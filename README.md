# reveal-deal-appeal

Intent prediction from a small, imbalanced dataset of transcripts 
by post-training a sentence transformer with contrastive learning, 
then training a classification head with labels.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
python src/refine.py
```

