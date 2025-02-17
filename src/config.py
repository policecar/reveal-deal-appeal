from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    max_length: int
    batch_size: int
    freeze_embeddings: bool


@dataclass
class TrainingConfig:
    learning_rate: float
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    weight_decay: float
    warmup_steps: int


@dataclass
class DataConfig:
    embeddings_path: str
    file_path: str
    finetune_dataset: str
    finetune_subset: Optional[str]
    train_split: float
    seed: int


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
        )
