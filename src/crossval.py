"""
Stratified k-fold cross-validation for the win/no-win classifier.

With a dataset this small, metrics from a single train/test split are too
noisy to compare experiments: a swing of two or three predictions moves F1
by several points. This harness trains a fresh model per fold and reports
macro-F1 and F2 (Win class, weighs recall higher) as mean +/- std across
folds. Results are also written to checkpoints/cv_metrics.json so runs can
be compared across experiments.

Besides SetFit it evaluates two cheap baselines on the identical folds:
TF-IDF + logistic regression and frozen (pretrained, not finetuned)
sentence-transformer embeddings + logistic regression. If SetFit cannot
beat these, the pipeline's complexity is not paying for itself.

Expects the preprocessed dataset at data/mauzo (run refine.py with
preprocess_data=True once to create it).

Usage:
    python src/crossval.py                   # baselines + SetFit
    python src/crossval.py --baselines-only  # skip the slow SetFit CV
"""

import argparse
import json

import numpy as np

from pathlib import Path
from datasets import concatenate_datasets, load_from_disk
from sentence_transformers import SentenceTransformer
from setfit import Trainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from config import Config
from refine import balanced_class_weights, build_model, build_training_args, get_device

N_SPLITS = 5


def fold_metrics(fold: int, y_true, y_pred) -> dict:
    return {
        "fold": fold,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f2_win": float(
            fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
        ),
    }


def summarize(name: str, folds: list[dict]) -> dict:
    macro = np.array([m["macro_f1"] for m in folds])
    f2 = np.array([m["f2_win"] for m in folds])
    print(
        f"{name:<20} macro-F1 {macro.mean():.3f} +/- {macro.std():.3f}   "
        f"F2 (Win) {f2.mean():.3f} +/- {f2.std():.3f}"
    )
    return {
        "macro_f1_mean": float(macro.mean()),
        "macro_f1_std": float(macro.std()),
        "f2_win_mean": float(f2.mean()),
        "f2_win_std": float(f2.std()),
        "folds": folds,
    }


def cross_validate_baseline(make_clf, X, labels, folds) -> list[dict]:
    """CV for an sklearn-style classifier; X is a list of texts (object
    array, so numpy fancy-indexing works) or an embedding matrix."""
    X = np.array(X, dtype=object) if not isinstance(X, np.ndarray) else X
    results = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        clf = make_clf()
        clf.fit(X[train_idx], labels[train_idx])
        y_pred = clf.predict(X[test_idx])
        results.append(fold_metrics(fold, labels[test_idx], y_pred))
    return results


def cross_validate_setfit(config: Config, data, folds) -> list[dict]:
    """Train a fresh SetFit model per fold."""
    device = get_device()
    num_classes = data.features["label"].num_classes

    results = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train_data = data.select(train_idx)
        test_data = data.select(test_idx)

        model = build_model(
            config,
            device,
            num_classes=num_classes,
            class_weights=balanced_class_weights(train_data["label"], num_classes),
        )
        trainer = Trainer(
            model=model,
            args=build_training_args(config, seed=config.data.seed + fold),
            train_dataset=train_data,
        )
        trainer.train()

        y_true = np.array(test_data["label"])
        y_pred = model.predict(test_data["text"]).cpu().numpy()
        results.append(fold_metrics(fold, y_true, y_pred))

        print(f"\nFold {fold}/{len(folds)}")
        # label 0 = no-win, 1 = win (see DatasetConverter._create_labels)
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["No-Win", "Win"],
                digits=3,
                zero_division=0,
            )
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="run only the cheap baselines, skip the SetFit CV",
    )
    args = parser.parse_args()

    config = Config.from_yaml("src/config.yaml")

    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir.parent / "data"

    # Pool the saved splits; CV makes its own train/test partitions per fold
    splits = load_from_disk(data_dir / "mauzo")
    data = concatenate_datasets([splits["train"], splits["test"]])
    texts = list(data["text"])
    labels = np.array(data["label"])

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=config.data.seed
    )
    folds = list(skf.split(np.zeros(len(labels)), labels))

    results = {}

    results["tfidf_logreg"] = cross_validate_baseline(
        lambda: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=config.data.seed,
            ),
        ),
        texts,
        labels,
        folds,
    )

    body = SentenceTransformer(config.model.name)
    body.max_seq_length = config.model.max_length
    embeddings = body.encode(
        texts, batch_size=config.model.batch_size, show_progress_bar=True
    )
    results["frozen_emb_logreg"] = cross_validate_baseline(
        lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=config.data.seed,
        ),
        embeddings,
        labels,
        folds,
    )

    if not args.baselines_only:
        results["setfit"] = cross_validate_setfit(config, data, folds)

    print(f"\n{'=' * 60}")
    print(f"Model: {config.model.name} @ {config.model.max_length} tokens, ")
    print(f"{N_SPLITS}-fold stratified CV, seed {config.data.seed}\n")
    summary = {
        "model": config.model.name,
        "max_length": config.model.max_length,
        "n_splits": N_SPLITS,
        "seed": config.data.seed,
        "models": {name: summarize(name, folds) for name, folds in results.items()},
    }

    results_path = script_dir.parent / "checkpoints" / "cv_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {results_path}")
