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

from pathlib import Path
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report

from config import Config
from data import DatasetConverter, DatasetAnonymizer
from plot import plot_embeddings_umap
# from utils import estimate_tokens

if __name__ == "__main__":
    preprocess_data = True
    train = True

    config = Config.from_yaml("src/config.yaml")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name = "BAAI/bge-small-en-v1.5"

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

    breakpoint()

    if train:
        # MODEL

        # from sentence_transformers import SentenceTransformer
        # from sklearn.linear_model import LogisticRegression
        # clf = LogisticRegression(
        #     class_weight="balanced",
        #     # max_iter=1000,
        #     # solver="liblinear",
        # )
        # model = SetFitModel(
        #     model_body=SentenceTransformer(model_name),
        #     model_head=clf
        # )

        model = SetFitModel.from_pretrained(
            model_name,
            use_differentiable_head=True,
            head_params={"out_features": len(train_data["label"])},
        )

        # TRAINING

        args = TrainingArguments(
            batch_size=8,  # (16, 2)
            # num_epochs=3,  # (1, 16)
            max_steps=100,
            # end_to_end=False,  # freeze body, train head
            l2_weight=0.1,  # 0.01
            sampling_strategy="undersampling",
            num_iterations=2,
            loss=CosineSimilarityLoss,  # default, consider FocalLoss
        )
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

    performance = classification_report(
        test_labels, y_pred, target_names=["Win", "No-Win"], digits=3
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
