import polars as pl

from datasets import Dataset
from typing import Dict, Optional


class DatasetConverter:
    def __init__(self, file_path: str, seed: int = 53):
        """Initialize from an excel file containing transcripts of sales calls."""
        self.sheet_name = "discovery calls"
        self.text_col = "Transcription"
        self.label_col = "Win/ No Win"

        self.seed = seed

        self.df = pl.read_excel(
            file_path,
            sheet_name=self.sheet_name,
        )
        self.df = self.df.filter(
            (pl.col(self.text_col).is_not_null())
            & (pl.col(self.text_col) != "")
            & (pl.col(self.text_col) != "null")
        )
        self.df = self.df.filter(
            (pl.col(self.label_col).is_not_null())
            & (pl.col(self.label_col) != "")
            & (pl.col(self.label_col) != "null")
        )
        print(self.df.get_column(self.label_col).value_counts())

    def _create_labels(self) -> list:
        """Create binary labels based on label column."""
        return [0 if dc == "No" else 1 for dc in self.df[self.label_col].to_list()]

    def to_dataset(self, train_split: Optional[float] = None) -> Dict[str, Dataset]:
        """
        Convert Polars DataFrame to HuggingFace Dataset format.

        Args:
            train_split: If provided, split the data into train/test sets.
                        Value should be between 0 and 1.

        Returns:
            Dict containing either {'train': Dataset, 'test': Dataset} if split,
            or {'train': Dataset} if no split.
        """
        # Prepare data
        texts = self.df["Transcription"].to_list()
        labels = self._create_labels()

        # Create dataset dictionary
        dataset_dict = {"text": texts, "label": labels}

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(dataset_dict)

        if train_split is not None:
            # Split into train/test
            split_dataset = dataset.train_test_split(
                train_size=train_split, seed=self.seed
            )
            return {"train": split_dataset["train"], "test": split_dataset["test"]}

        return {"train": dataset}
