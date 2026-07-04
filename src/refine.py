"""
Sentence transformer finetuning

A script using SeTFit to few-shot finetune a sentence transformer model
with a classification head for a binary classification task.

The model consists of
- a sentence transformer embedding base
- a classifier head, e.g., logistic regression or a neural network

Training happens in two phases:
- Finetuning embeddings with constrastive learning
  based on positive and negative pairs of sentences
- Training a classification head
  based on embedded sentences and their labels

See also:
- https://huggingface.co/docs/setfit/en/conceptual_guides/setfit


Teuxdeux:
- augment text (e.g., with nlpaug)
- add a bottleneck between embedding and classifier
"""

import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report

from config import Config
from clf import BottleneckClassifier
from dataset import DatasetConverter, DatasetAnonymizer
from plot import plot_embeddings_umap
# from utils import estimate_tokens


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def build_model(
    config: Config, device: torch.device, num_classes: int = 2
) -> SetFitModel:
    """
    Build a SetFit model from the config: a sentence transformer body
    (plain encoders like ModernBERT get mean pooling added automatically)
    with a bottleneck classification head.
    """
    model_body = SentenceTransformer(config.model.name)
    model_body.max_seq_length = config.model.max_length

    # Trade ~30% compute for an order of magnitude less activation memory.
    # Without this, contrastive training at 1024 tokens overruns 24GB unified
    # memory and MPS swap-thrashes (~280s/step instead of seconds).
    if config.model.gradient_checkpointing:
        model_body[0].auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # in_features must be explicit: SetFitHead.__init__ xavier-initializes
    # every nn.Linear, and the LazyLinear used when in_features=None is an
    # nn.Linear subclass whose uninitialized weight crashes that init.
    clf = BottleneckClassifier(
        in_features=model_body.get_sentence_embedding_dimension(),
        bottleneck_dim=config.model.bottleneck_dim,
        out_features=num_classes,
        device=device,
    )
    return SetFitModel(
        model_body=model_body,
        model_head=clf,
        use_differentiable_head=True,
    )


def build_training_args(config: Config, seed: int) -> TrainingArguments:
    return TrainingArguments(
        batch_size=config.model.batch_size,  # pairs per step; 2 sequences each
        # num_epochs=3,  # (1, 16)
        max_steps=100,
        # end_to_end=False,  # freeze body, train head
        l2_weight=0.1,  # 0.01
        sampling_strategy="undersampling",
        num_iterations=2,
        loss=CosineSimilarityLoss,  # default, consider FocalLoss
        seed=seed,
    )


if __name__ == "__main__":
    preprocess_data = False
    train = True

    config = Config.from_yaml("src/config.yaml")
    model_name = config.model.name

    device = get_device()

    script_dir = Path(__file__).parent.absolute()
    ckpt_dir = script_dir.parent / "checkpoints"
    data_dir = script_dir.parent / "data"

    # DATA

    if preprocess_data:
        converter = DatasetConverter(config.data)
        data = converter.to_dataset(
            config.data.train_split, shuffle=config.data.shuffle
        )

        anonymizer = DatasetAnonymizer()
        data = anonymizer.anonymize_dataset(data, text_column="text")

        # save data
        data.save_to_disk(data_dir / "mauzo")
    else:
        # load preprocessed data
        from datasets import load_from_disk

        data = load_from_disk(data_dir / "mauzo")

    train_data = data["train"]
    test_data = data["test"]

    # sample from training data
    # from setfit import sample_dataset
    # train_data = sample_dataset(train_data, label_column="label", num_samples=8)

    if train:
        # MODEL

        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression(
        #     class_weight="balanced",
        #     # max_iter=1000,
        #     # solver="liblinear",
        # )

        model = build_model(
            config, device, num_classes=train_data.features["label"].num_classes
        )

        # TRAINING

        args = build_training_args(config, seed=config.data.seed)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=test_data,
            metric="accuracy",
        )
        trainer.train()

        model.save_pretrained(ckpt_dir / f"{model_name.split('/')[-1]}_setfit")

        metrics = trainer.evaluate(test_data)
        print(metrics)

    else:
        model = SetFitModel.from_pretrained(
            ckpt_dir / f"{model_name.split('/')[-1]}_setfit"
        )
        # model = SetFitModel.from_pretrained(model_name)

    # EVALUATION

    train_labels = np.array(train_data["label"])
    test_labels = np.array(test_data["label"])

    y_pred = model.predict(test_data["text"]).cpu().numpy()

    # label 0 = no-win, 1 = win (see DatasetConverter._create_labels)
    performance = classification_report(
        test_labels, y_pred, target_names=["No-Win", "Win"], digits=3
    )
    print(f"\n{performance}")

    # from sklearn.metrics import fbeta_score
    # # F2 score (weighs recall higher)
    # f2 = fbeta_score(test_labels, y_pred, beta=2, pos_label=1)
    # print(f"\nF2 score (Win class): {f2:.4f}")

    # VISUALIZATION

    # Embed data, run UMAP, then plot the projected embeddings
    train_embeddings = model.model_body.encode(train_data["text"])
    plot_embeddings_umap(train_embeddings, train_labels, idxs=train_data["index"])

    test_embeddings = model.model_body.encode(test_data["text"])
    plot_embeddings_umap(test_embeddings, test_labels, idxs=test_data["index"])

    # embeddings = np.vstack([train_embeddings, test_embeddings])
    # labels = np.concatenate([train_labels, test_labels])
    # idxs = np.concatenate([train_data["index"], test_data["index"]])
    # plot_embeddings_umap(embeddings, labels, idxs=idxs)

    breakpoint()
