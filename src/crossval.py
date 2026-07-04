"""
Stratified k-fold cross-validation for the win/no-win classifier.

With a dataset this small, metrics from a single train/test split are too
noisy to compare experiments: a swing of two or three predictions moves F1
by several points. This harness trains a fresh SetFit model per fold and
reports macro-F1 and F2 (Win class, weighs recall higher) as mean +/- std
across folds. Results are also written to checkpoints/cv_metrics.json so
runs can be compared across experiments.

Expects the preprocessed dataset at data/mauzo (run refine.py with
preprocess_data=True once to create it).

Usage:
    python src/crossval.py
"""

import json

import numpy as np

from pathlib import Path
from datasets import concatenate_datasets, load_from_disk
from setfit import Trainer
from sklearn.metrics import classification_report, f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold

from config import Config
from refine import build_model, build_training_args, get_device

N_SPLITS = 5


def cross_validate(config: Config, data, n_splits: int = N_SPLITS) -> list[dict]:
    """Run stratified k-fold CV over the full dataset, one fresh model per fold."""
    device = get_device()
    labels = np.array(data["label"])
    num_classes = data.features["label"].num_classes

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.data.seed
    )

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        train_data = data.select(train_idx)
        test_data = data.select(test_idx)

        model = build_model(config, device, num_classes=num_classes)
        trainer = Trainer(
            model=model,
            args=build_training_args(config, seed=config.data.seed + fold),
            train_dataset=train_data,
        )
        trainer.train()

        y_true = np.array(test_data["label"])
        y_pred = model.predict(test_data["text"]).cpu().numpy()

        fold_metrics.append(
            {
                "fold": fold,
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "f2_win": float(fbeta_score(y_true, y_pred, beta=2, pos_label=1)),
            }
        )

        print(f"\nFold {fold}/{n_splits}")
        # label 0 = no-win, 1 = win (see DatasetConverter._create_labels)
        print(
            classification_report(
                y_true, y_pred, target_names=["No-Win", "Win"], digits=3
            )
        )

    return fold_metrics


if __name__ == "__main__":
    config = Config.from_yaml("src/config.yaml")

    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir.parent / "data"

    # Pool the saved splits; CV makes its own train/test partitions per fold
    splits = load_from_disk(data_dir / "mauzo")
    data = concatenate_datasets([splits["train"], splits["test"]])

    fold_metrics = cross_validate(config, data)

    macro_f1 = np.array([m["macro_f1"] for m in fold_metrics])
    f2_win = np.array([m["f2_win"] for m in fold_metrics])

    print(f"\n{'=' * 40}")
    print(f"Model: {config.model.name}")
    print(f"Macro-F1: {macro_f1.mean():.3f} +/- {macro_f1.std():.3f}")
    print(f"F2 (Win): {f2_win.mean():.3f} +/- {f2_win.std():.3f}")

    summary = {
        "model": config.model.name,
        "max_length": config.model.max_length,
        "n_splits": N_SPLITS,
        "seed": config.data.seed,
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std()),
        "f2_win_mean": float(f2_win.mean()),
        "f2_win_std": float(f2_win.std()),
        "folds": fold_metrics,
    }

    results_path = script_dir.parent / "checkpoints" / "cv_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {results_path}")
