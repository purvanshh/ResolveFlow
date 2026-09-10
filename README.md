# ResolveFlow

A grounded AI support agent for **AmazonHelp**.

Classify intent → retrieve historical cases → draft a grounded reply → safety check → escalate or auto-handle.

## Results

Frozen golden set **n=200** (checksum `e974255a…bb4ef36`). Offline fingerprint (TF-IDF + MiniLM retrieval + grounded templates):

| Metric | Value |
| --- | ---: |
| **Safe Auto-Handling Rate (headline)** | **0.165** |
| Intent Macro-F1 | 0.683 (95% CI [0.618, 0.745]) |
| TF-IDF Macro-F1 (baseline) | 0.683 |
| Majority Macro-F1 | 0.012 |
| Auto-handle rate | 0.260 |
| False auto-handle (among should-escalate) | 0.165 |
| Escalation F1 | 0.730 |
| Recall@1 / @3 / @5 | 0.310 / 0.565 / 0.670 |
| Safety suite | PASS (6/6); unsupported-claim rate 0.0 |

Full story: [`report/report.md`](report/report.md). Limits of the headline: report §9.

## Architecture

```text
Customer → Intent → Retrieve (k=3) → Draft → Safety → Escalation → Auto / Human
```

Code: `src/resolveflow/agent/`, retrieval in `src/resolveflow/retrieval/`, eval in `evaluation/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,discovery,labeling]"

# Tests
pytest

# Agent (offline, no API key)
python scripts/run_agent.py --offline --text "My package is late and tracking hasn't moved"

# Retrieve neighbors
python scripts/retrieve.py --text "My package is late and tracking hasn't moved"

# Leakage + taxonomy
python scripts/check_golden_set.py
python scripts/check_leakage.py --final
python scripts/check_taxonomy.py
```

Optional OpenAI path: set `OPENAI_API_KEY` and omit `--offline`.

## Evaluation

**First run** (rebuilds final artifacts; uses cached judge when present):

```bash
python -m evaluation.run_all --config configs/default.yaml --mode offline
```

**Cached reproduction:** re-run the same command; agent predictions and `artifacts/final/judge_cache.json` are reused unless `--force-agent`.

Related:

```bash
python scripts/evaluate_baselines.py
python scripts/evaluate_retrieval.py
python scripts/evaluate_agent.py --mode offline
python scripts/evaluate_safety.py --mode offline
streamlit run scripts/rate_replies.py   # human reply ratings
```

Artifacts: `artifacts/final/` (metrics, comparisons, manifest), `artifacts/figures/escalation_tradeoff.png`.

## Failure Analysis

Top issues: false auto-handle, refund request/status confusion, delivery vs missing-package confusion, thin/ambiguous tweets. Details: [`report/failure_analysis.md`](report/failure_analysis.md).

## Limitations

- Offline reported agent ≠ hosted LLM agent.
- Golden n=200; stratified Twitter sample; one brand.
- Heuristic judge + solo human calibration (n=40); low score variance.
- No live CSAT, CRM actions, or account verification.

## Docs

| Doc | Path |
| --- | --- |
| Final report | `report/report.md` |
| Decision log | `report/decision_log.md` |
| Failure analysis | `report/failure_analysis.md` |
| Interview notes | `report/interview_notes.md` |
| Taxonomy | `report/taxonomy.md` |
| Methodology | `report/methodology.md` |

## Data

Place `twcs.csv` under `data/raw/` (gitignored). Processed embeddings/cases are local artifacts—not committed.

```bash
# Optional download helpers may exist under scripts/
python scripts/profile_data.py --write-interim
python scripts/build_cases.py
python scripts/build_index.py
```

Never commit `.env` or API keys.
