# Golden evaluation set

This directory contains the **frozen human-labelled evaluation set** for ResolveFlow.

The set must **not** be used for:

- model training
- prompt tuning
- retrieval / few-shot examples
- threshold tuning
- taxonomy development after freeze
- any development that would leak labels into the system under test

It is reserved for **final evaluation** only.

## Files

| File | Purpose |
| --- | --- |
| `golden_set.csv` | Ground-truth examples |
| `golden_set_manifest.json` | Freeze status + SHA-256 |
| `ANNOTATION_GUIDE.md` | Labeling rules |
| `.label_progress.json` | Streamlit resume pointer (local) |

## Commands

```bash
streamlit run scripts/label_golden_set.py
python scripts/check_taxonomy.py
python scripts/freeze_golden_set.py
python scripts/check_leakage.py --build-dev
python scripts/golden_set_stats.py
```

After freeze, treat checksum mismatches as a new version — do not silent-edit.
