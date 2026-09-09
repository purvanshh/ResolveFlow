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

## Phase 2 — Taxonomy + golden set

```bash
pip install -e ".[dev,discovery,labeling]"
python scripts/discover_intents.py
streamlit run scripts/label_golden_set.py   # optional review UI
python scripts/check_taxonomy.py
python scripts/freeze_golden_set.py
python scripts/check_leakage.py --build-dev
python scripts/golden_set_stats.py
```

See `configs/intents.yaml`, `data/golden/`, and `report/taxonomy.md`.

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
