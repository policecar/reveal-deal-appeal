"""Regenerate data/mauzo — mirrors refine.py's preprocess_data=True path.

Run from the repo root: python src/preprocess.py
"""

from config import Config
from dataset import DatasetConverter, DatasetAnonymizer

config = Config.from_yaml("src/config.yaml")

converter = DatasetConverter(config.data)
data = converter.to_dataset(config.data.train_split, shuffle=config.data.shuffle)

anonymizer = DatasetAnonymizer()
data = anonymizer.anonymize_dataset(data, text_column="text")

data.save_to_disk("data/mauzo")
print("\nSaved to data/mauzo")
for split, ds in data.items():
    labels = ds["label"]
    print(
        f"{split}: {len(ds)} rows, wins={sum(labels)}, no-wins={len(labels) - sum(labels)}"
    )
