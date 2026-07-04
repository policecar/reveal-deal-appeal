"""
Stratified k-fold cross-validation for the win/no-win classifier.

With a dataset this small, metrics from a single train/test split are too
noisy to compare experiments: a swing of two or three predictions moves F1
by several points. This harness trains a fresh model per fold and reports
macro-F1, F2 (Win class, weighs recall higher) and AUROC as mean +/- std
across folds, plus a pooled AUROC over all out-of-fold scores and the rank
of every win call when the corpus is sorted by predicted win probability.
Results are written to checkpoints/cv_metrics.json.

Besides SetFit it evaluates cheap baselines on the identical folds:
- TF-IDF + logistic regression
- frozen (pretrained, not finetuned) embeddings of the FIRST max_length
  tokens + logistic regression
- chunked frozen embeddings covering the WHOLE call (median call is ~10x
  longer than max_length), mean-pooled per call, + logistic regression
- the same chunks scored individually, call score = max chunk probability
  ("did any part of this call sound like a win?")

If SetFit cannot beat these, the pipeline's complexity is not paying for
itself.

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
from sklearn.metrics import (
    classification_report,
    f1_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from config import Config
from refine import balanced_class_weights, build_model, build_training_args, get_device

N_SPLITS = 5
CHUNK_OVERLAP = 128


def fold_metrics(fold: int, y_true, y_pred, y_score=None) -> dict:
    m = {
        "fold": fold,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f2_win": float(
            fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
        ),
    }
    if y_score is not None:
        m["auroc"] = float(roc_auc_score(y_true, y_score))
    return m


def summarize(name: str, result: dict, labels) -> dict:
    folds = result["folds"]
    out = {"folds": folds}
    line = f"{name:<22}"
    for key, label in [
        ("macro_f1", "macro-F1"),
        ("f2_win", "F2(Win)"),
        ("auroc", "AUROC"),
    ]:
        vals = [m[key] for m in folds if key in m]
        if vals:
            vals = np.array(vals)
            out[f"{key}_mean"], out[f"{key}_std"] = (
                float(vals.mean()),
                float(vals.std()),
            )
            line += f"  {label} {vals.mean():.3f}+/-{vals.std():.3f}"

    scores = result.get("oof_scores")
    if scores is not None and not np.isnan(scores).any():
        out["oof_auroc"] = float(roc_auc_score(labels, scores))
        # 1-based rank of every win when all calls are sorted by score, best first
        order = np.argsort(-scores)
        ranks = sorted(
            int(np.where(order == i)[0][0]) + 1 for i in np.flatnonzero(labels == 1)
        )
        out["win_ranks"] = ranks
        line += f"  | pooled AUROC {out['oof_auroc']:.3f}, win ranks {ranks} of {len(labels)}"

    print(line)
    return out


def cross_validate_baseline(make_clf, X, labels, folds) -> dict:
    """CV for an sklearn-style classifier; X is a list of texts (object
    array, so numpy fancy-indexing works) or an embedding matrix."""
    X = np.array(X, dtype=object) if not isinstance(X, np.ndarray) else X
    results, oof = [], np.full(len(labels), np.nan)
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        clf = make_clf()
        clf.fit(X[train_idx], labels[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_score = clf.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = y_score
        results.append(fold_metrics(fold, labels[test_idx], y_pred, y_score))
    return {"folds": results, "oof_scores": oof}


def chunk_token_ids(text: str, tokenizer, chunk_tokens: int, overlap: int) -> list[str]:
    """Split a text into decoded windows of chunk_tokens tokens with overlap,
    so the encoder sees the whole call instead of its first minutes."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    stride = chunk_tokens - overlap
    starts = range(0, max(len(ids) - overlap, 1), stride)
    return [tokenizer.decode(ids[s : s + chunk_tokens]) for s in starts]


def encode_chunks(body, texts, chunk_tokens: int, overlap: int, batch_size: int = 8):
    """Embed every chunk of every text. Returns (chunk_embeddings, owners)
    where owners[j] is the index of the call chunk j belongs to."""
    chunks, owners = [], []
    for i, text in enumerate(texts):
        pieces = chunk_token_ids(text, body.tokenizer, chunk_tokens, overlap)
        chunks.extend(pieces)
        owners.extend([i] * len(pieces))
    print(f"encoding {len(chunks)} chunks for {len(texts)} calls")
    embeddings = body.encode(chunks, batch_size=batch_size, show_progress_bar=True)
    return np.asarray(embeddings), np.array(owners)


def mean_pool_chunks(chunk_emb, owners, n_calls: int):
    return np.vstack([chunk_emb[owners == i].mean(axis=0) for i in range(n_calls)])


def cross_validate_chunk_maxprob(make_clf, chunk_emb, owners, labels, folds) -> dict:
    """Train on chunks (each inherits its call's label); score a call by the
    max win probability over its chunks."""
    results, oof = [], np.full(len(labels), np.nan)
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train_mask = np.isin(owners, train_idx)
        clf = make_clf()
        clf.fit(chunk_emb[train_mask], labels[owners[train_mask]])
        y_score = np.array(
            [clf.predict_proba(chunk_emb[owners == i])[:, 1].max() for i in test_idx]
        )
        y_pred = (y_score >= 0.5).astype(int)
        oof[test_idx] = y_score
        results.append(fold_metrics(fold, labels[test_idx], y_pred, y_score))
    return {"folds": results, "oof_scores": oof}


def cross_validate_setfit(config: Config, data, folds) -> dict:
    """Train a fresh SetFit model per fold."""
    device = get_device()
    num_classes = data.features["label"].num_classes
    labels = np.array(data["label"])

    results, oof = [], np.full(len(labels), np.nan)
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
        y_score = model.predict_proba(test_data["text"])[:, 1].cpu().numpy()
        oof[test_idx] = y_score
        results.append(fold_metrics(fold, y_true, y_pred, y_score))

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

    return {"folds": results, "oof_scores": oof}


def make_logreg(seed: int):
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="run only the cheap baselines, skip the SetFit CV",
    )
    args = parser.parse_args()

    config = Config.from_yaml("src/config.yaml")
    seed = config.data.seed

    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir.parent / "data"

    # Pool the saved splits; CV makes its own train/test partitions per fold
    splits = load_from_disk(data_dir / "mauzo")
    data = concatenate_datasets([splits["train"], splits["test"]])
    texts = list(data["text"])
    labels = np.array(data["label"])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(labels)), labels))

    results = {}

    results["tfidf_logreg"] = cross_validate_baseline(
        lambda: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
            make_logreg(seed),
        ),
        texts,
        labels,
        folds,
    )

    body = SentenceTransformer(config.model.name)
    body.max_seq_length = config.model.max_length

    # truncated view: first max_length tokens only (what SetFit sees)
    embeddings = body.encode(
        texts, batch_size=config.model.batch_size, show_progress_bar=True
    )
    results["frozen_emb_logreg"] = cross_validate_baseline(
        lambda: make_logreg(seed), embeddings, labels, folds
    )

    # whole-call view: chunk, embed everything, pool
    chunk_emb, owners = encode_chunks(
        body,
        texts,
        chunk_tokens=config.model.max_length - 16,
        overlap=CHUNK_OVERLAP,
    )
    pooled = mean_pool_chunks(chunk_emb, owners, len(texts))
    results["chunked_mean_logreg"] = cross_validate_baseline(
        lambda: make_logreg(seed), pooled, labels, folds
    )
    results["chunked_maxprob_logreg"] = cross_validate_chunk_maxprob(
        lambda: make_logreg(seed), chunk_emb, owners, labels, folds
    )

    if not args.baselines_only:
        results["setfit"] = cross_validate_setfit(config, data, folds)

    print(f"\n{'=' * 76}")
    print(
        f"Model: {config.model.name} @ {config.model.max_length} tokens, "
        f"{N_SPLITS}-fold stratified CV, seed {seed}\n"
    )
    summary = {
        "model": config.model.name,
        "max_length": config.model.max_length,
        "n_splits": N_SPLITS,
        "seed": seed,
        "models": {
            name: summarize(name, result, labels) for name, result in results.items()
        },
    }

    results_path = script_dir.parent / "checkpoints" / "cv_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {results_path}")
