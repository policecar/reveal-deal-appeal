import multiprocessing
import numpy as np
import pandas as pd

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from config import Config
from datasets import Dataset, DatasetDict
from typing import Dict, Optional


class DatasetConverter:
    def __init__(self, config: Config):
        """Initialize from an excel file containing transcripts of sales calls."""
        self.seed = config.seed
        self.text_col = config.excel_text_col
        self.label_col = config.excel_label_col

        self.df = pd.read_excel(
            config.file_path,
            sheet_name=config.excel_sheet_name,
        )
        # Store original indices and dates before any filtering
        original_indices = set(self.df.index)
        dates_dict = self.df["Date"].to_dict()

        (self._apply_basic_filters()._filter_length_outliers())

        # Print skipped indices using stored dates
        skipped_indices = list(original_indices - set(self.df.index))
        if skipped_indices:
            # Find the width needed for the largest index
            max_idx_width = len(str(max(skipped_indices)))
            print("\nSkipped rows (index, date):")
            for idx in sorted(skipped_indices):
                date_str = dates_dict[idx].strftime("%Y-%m-%d")
                print(f"{idx:>{max_idx_width}}: {date_str}")

        # Reset index and store original indices in a column
        self.df = self.df.reset_index(names=["original_index"])

        print(self.df[self.label_col].value_counts())

    def _apply_basic_filters(self) -> "DatasetConverter":
        """Apply basic null and length filters."""
        self.df = self.df[
            (self.df[self.text_col].notna())
            & (self.df[self.text_col] != "")
            & (self.df[self.text_col] != "null")
            & (self.df[self.label_col].notna())
            & (self.df[self.label_col] != "")
            & (self.df[self.label_col] != "null")
        ].copy()
        return self

    def _filter_length_outliers(self, z_threshold: float = 2.0) -> "DatasetConverter":
        """Filter out outliers based on text length statistics."""
        # Add character and token count columns
        self.df["char_count"] = self.df[self.text_col].str.len()
        self.df["token_count"] = self.df[self.text_col].str.split().str.len()

        # Calculate z-scores
        char_zscore = np.abs(
            (self.df["char_count"] - self.df["char_count"].mean())
            / self.df["char_count"].std()
        )
        token_zscore = np.abs(
            (self.df["token_count"] - self.df["token_count"].mean())
            / self.df["token_count"].std()
        )

        print(
            f"\nCharacter count mean: {self.df['char_count'].mean():.1f}, std: {self.df['char_count'].std():.1f}"
        )
        print(
            f"Token count mean: {self.df['token_count'].mean():.1f}, std: {self.df['token_count'].std():.1f}"
        )

        # Apply filters
        self.df = self.df[
            (char_zscore <= z_threshold) & (token_zscore <= z_threshold)
        ].drop(columns=["char_count", "token_count"])

        return self

    def to_dataset(
        self, train_split: Optional[float] = None, shuffle: bool = True
    ) -> Dict[str, Dataset]:
        """
        Convert Pandas DataFrame to HuggingFace Dataset format.

        Args:
            train_split: If provided, split the data into train/test sets.
                        Value should be between 0 and 1.

        Returns:
            Dict containing either {'train': Dataset, 'test': Dataset} if split,
            or {'train': Dataset} if no split.
        """
        # Prepare data
        texts = self.df[self.text_col].tolist()
        labels = self._create_labels()
        original_indices = self.df.index.tolist()

        # Create dataset dictionary
        dataset_dict = {"text": texts, "label": labels, "index": original_indices}

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(dataset_dict)

        if train_split is not None:
            # Split into train/test
            split_dataset = dataset.train_test_split(
                train_size=train_split,
                seed=self.seed,
                shuffle=shuffle,
            )
            return {"train": split_dataset["train"], "test": split_dataset["test"]}

        return {"train": dataset}

    def _create_labels(self) -> list:
        """Create binary labels based on label column."""
        return [0 if dc == "No" else 1 for dc in self.df[self.label_col]]


class DatasetAnonymizer:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def _clean_text(self, text: str) -> str:
        analyzer_results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=["PERSON", "ORGANIZATION", "LOCATION"],
        )
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<person>"}),
            "ORGANIZATION": OperatorConfig("replace", {"new_value": "<org>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<location>"}),
        }
        anonymized_text = self.anonymizer.anonymize(
            text=text, analyzer_results=analyzer_results, operators=operators
        )
        return anonymized_text.text.lower()

    def process_split(
        self,
        split_data: Dataset,
        text_column: str,
        batch_size: int = 1000,
    ) -> Dataset:
        num_proc = max(1, multiprocessing.cpu_count() - 1)

        def process_batch(examples):
            return {
                text_column: [self._clean_text(text) for text in examples[text_column]]
            }

        return split_data.map(
            process_batch,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Anonymizing data",
        )

    def anonymize_dataset(
        self,
        data: dict,
        text_column: str,
        batch_size: int = 100,
    ):
        anonymized_splits = {}
        for split_name, split_data in data.items():
            anonymized_splits[split_name] = self.process_split(
                split_data=split_data,
                text_column=text_column,
                batch_size=batch_size,
            )

        return DatasetDict(anonymized_splits)
