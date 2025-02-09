"""
Training a classifier for purchase intent detection in sales calls.

Plan:
* ⁠Baseline 1: TF-IDF plus logistic regression.
* ⁠⁠Baseline 2: Prompt LLM with transcript and ask if buy or no-buy (optional mit prompt optimization).
* Feature extraction via embedding model plus simple classifier.
* Finetune a foundation model (e.g., a xxBERTxxx) – either SFT or using few-shot learning (e.g., with SetFit)

v1: Predict purchase intent from first 512 tokens per sample using a task-specific model.

v2: Generate embeddings of lenght 4096 by mean pooling over each sample,
    then train a logistic regression classifier to predict purchase intent.

v3:

"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import Config
from data import DatasetConverter


def get_embeddings(
    texts, model, tokenizer, config: Config, device="cuda", batch_size=4
):
    """Generate embeddings for a list of texts using the specified model.

    Args:
        texts: List of texts to embed
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        device: Computing device
        batch_size: Number of texts to process at once (adjust based on available RAM)
    """
    embeddings = []

    # Process texts in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            # Get batch of texts
            batch_texts = texts[i : i + batch_size]

            # Tokenize and prepare input
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.model.max_length,
                return_tensors="pt",
            ).to(device)

            # Get model outputs
            outputs = model(**inputs)

            # Mean pooling over non-padding tokens
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state

            # More memory-efficient mask expansion
            input_mask_expanded = attention_mask.unsqueeze(-1)

            # Compute mean embeddings for batch
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9).unsqueeze(-1)
            batch_embeddings = sum_embeddings / sum_mask

            # Move to CPU and convert to numpy
            embeddings.append(batch_embeddings.cpu().numpy())

            # Clear memory
            del inputs, outputs, token_embeddings, attention_mask
            if device == "cuda":
                torch.cuda.empty_cache()

    return np.vstack(embeddings)


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    config = Config.from_yaml("config.yaml")

    # DATA

    converter = DatasetConverter(config.data.file_path)
    data = converter.to_dataset(train_split=0.8)

    # MODEL

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=False)
    model = AutoModel.from_pretrained(config.model.name).to(device)

    # Generate embeddings
    train_embeddings = get_embeddings(
        data["train"]["text"], model, tokenizer, config, device, batch_size=16
    )
    test_embeddings = get_embeddings(
        data["test"]["text"], model, tokenizer, config, device, batch_size=16
    )

    # TRAINING

    train_labels = data["train"]["label"]

    # Calibrated classifier
    base_classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
    classifier = CalibratedClassifierCV(
        base_classifier,
        cv=5,  # 5-fold cross-validation
        method="sigmoid",
    )
    classifier.fit(train_embeddings, train_labels)

    # INFERENCE

    y_pred = classifier.predict(test_embeddings)
    y_pred_proba = classifier.predict_proba(test_embeddings)

    # EVALUATION

    y_true = data["test"]["label"]
    performance = classification_report(y_true, y_pred, target_names=["No-Buy", "Buy"])
    print(performance)

    # VISUALIZATION

    # positions = [i for i in range(len(y_pred))]

    # Plot violin plot of scores
    # plt.violinplot([scores_neg, scores_pos], showextrema=True)
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.violinplot([y_pred_proba[:, 0], y_pred_proba[:, 1]], showextrema=True)
    plt.xticks([1, 2], ["No-Buy", "Buy"])
    plt.ylabel("Probability")
    plt.title("Distribution of Prediction Probabilities")
    plt.show()

    # breakpoint()
