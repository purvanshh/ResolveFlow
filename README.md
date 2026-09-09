# ResolveFlow

Phase-driven customer support resolution pipeline built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (`thoughtvector/customer-support-on-twitter`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset

Place the raw CSV under `data/raw/` (typically `twcs.csv`).
Raw / interim / processed data are gitignored.

```bash
# Optional: download via Kaggle CLI
# kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
```

## Phase 1 — Data profile

```bash
python scripts/download_dataset.py   # if twcs.csv is missing
python scripts/profile_data.py --write-interim
```

Selected brand for later phases: **AmazonHelp** (see `report/methodology.md`).

## Tests

```bash
pytest
```

## Project layout

```text
resolveflow/
├── configs/
├── data/{raw,interim,processed}/
├── report/
├── scripts/
├── src/resolveflow/
└── tests/
```
