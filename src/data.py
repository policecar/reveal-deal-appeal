import multiprocessing
import polars as pl

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

        self.df = pl.read_excel(
            config.file_path,
            sheet_name=config.excel_sheet_name,
        )
        self.df = self.df.filter(
            (pl.col(self.text_col).is_not_null())
            & (pl.col(self.text_col) != "")
            & (pl.col(self.text_col) != "null")
            & (pl.col(self.text_col).str.len_chars() >= 100)
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

    def to_dataset(
        self, train_split: Optional[float] = None, shuffle: bool = True
    ) -> Dict[str, Dataset]:
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
        # Add original indices to the dataset
        original_indices = pl.arange(0, len(self.df), eager=True).to_list()

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
