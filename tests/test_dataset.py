"""Unit tests for DatasetConverter: label mapping, ClassLabel order, index tracking.

Uses a small synthetic Excel file; no ML dependencies are exercised.
"""

import pandas as pd

from config import DataConfig
from dataset import DatasetConverter

TEXT_COL = "Transcription"
LABEL_COL = "Win/ No Win"
SHEET = "discovery calls"


def make_df(labels: list, texts: list | None = None) -> pd.DataFrame:
    n = len(labels)
    if texts is None:
        # Varied lengths: identical lengths make the char/token std zero and
        # the outlier z-score filter degenerate (NaN drops every row).
        texts = [f"call transcript {i} " + "blah " * (10 + i) for i in range(n)]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01"] * n),
            TEXT_COL: texts,
            LABEL_COL: labels,
        }
    )


def make_config(tmp_path, df: pd.DataFrame) -> DataConfig:
    path = tmp_path / "transcripts.xlsx"
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name=SHEET, index=False)
    return DataConfig(
        file_path=str(path),
        excel_sheet_name=SHEET,
        excel_text_col=TEXT_COL,
        excel_label_col=LABEL_COL,
        embeddings_path="",
        finetune_dataset="",
        finetune_subset=None,
        train_split=0.8,
        seed=7,
        shuffle=True,
    )


def test_label_mapping(tmp_path):
    """Exactly 'No' maps to 0; every other value currently maps to 1.

    Documents current behavior, including that 'Potential Win' and
    'Win - MVP never started' count as wins — flagged as a known issue
    (normalize + raise on unexpected values).
    """
    labels = ["No", "Win", "Potential Win", "Win - MVP never started", "No", "No"]
    converter = DatasetConverter(make_config(tmp_path, make_df(labels)))
    ds = converter.to_dataset()["train"]
    assert ds["label"] == [0, 1, 1, 1, 0, 0]


def test_class_label_order_independent_of_data_order(tmp_path):
    """0 = no-win, 1 = win even when a Win row comes first in the file."""
    labels = ["Win", "No", "No", "Win", "No", "No"]
    converter = DatasetConverter(make_config(tmp_path, make_df(labels)))
    ds = converter.to_dataset()["train"]

    assert ds.features["label"].names == ["no-win", "win"]
    assert ds.features["label"].str2int("win") == 1
    assert ds["label"] == [1, 0, 0, 1, 0, 0]


def test_original_index_survives_filtering(tmp_path):
    """Rows dropped by filters must not shift the indices of surviving rows."""
    labels = ["No", "Win", "No", "No", "Win", "No"]
    df = make_df(labels)
    df.loc[1, TEXT_COL] = ""  # dropped by the basic filters

    converter = DatasetConverter(make_config(tmp_path, df))
    ds = converter.to_dataset()["train"]

    assert ds["index"] == [0, 2, 3, 4, 5]
    assert ds["label"] == [0, 0, 0, 1, 0]
