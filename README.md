# ResolveFlow

A grounded AI support agent for **AmazonHelp**.

Classify intent → retrieve historical cases → draft a grounded reply → safety check → escalate or auto-handle.

## Results

Frozen golden set **n=200** (checksum `e974255a…bb4ef36`).  
**Fingerprint:** `gpt-4o-mini` classifier + responder + MiniLM retrieval (k=3) + risk-aware escalation + `gpt-4o-mini` judge.

**Safe Auto-Handling Rate** = fraction of golden examples that were auto-handled **and** had correct intent **and** no safety/unsupported-claim flags **and** gold also auto_handle (**51/200 = 0.255**).  
**Auto-handle rate (0.440)** is only “policy did not escalate”—not the same as safe automation.

| Metric | Value |
| --- | ---: |
| **Safe Auto-Handling Rate (headline)** | **0.255** |
| Intent Macro-F1 (ResolveFlow / LLM) | 0.669 (95% CI [0.598, 0.727]) |
| TF-IDF Macro-F1 (baseline) | **0.683** |
| Majority Macro-F1 | 0.012 |
| Auto-handle rate | 0.440 |
| False auto-handle (among should-escalate) | 0.243 |
| Escalation F1 | 0.767 |
| Recall@1 / @3 / @5 | 0.310 / 0.565 / 0.670 |
| Reply correctness / groundedness (LLM judge) | 4.36 / 4.33 |
| Unsupported-claim rate (drafts) | 0.015 |
| Safety suite | **6/6 PASS** (GPT `account_specific` + DeepSeek `fake_policy` were harness FPs → regraded on stored replies) |

On this golden set, **TF-IDF still edges the LLM on intent Macro-F1**; ResolveFlow’s gains show up in **reply quality vs generic/nearest baselines** and a higher **safe auto-handle rate** than the offline template agent (0.255 vs 0.165), with escalation F1 0.767.

Full story: [`report/report.md`](report/report.md). Limits of the headline: report §9.

## LLM Model Comparison

Controlled substitution on the same frozen golden set (n=200), retrieval (k=3), safety, escalation, and **gpt-4o-mini judge**. DeepSeek API model: `deepseek-flash` (version `DeepSeek-V4.1-Flash`).

| Metric | TF-IDF | GPT-4o-mini | DeepSeek Non-thinking | DeepSeek Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | **0.683** | 0.669 | 0.653 | 0.634 |
| Auto-handle rate | — | 0.440 | 0.470 | 0.500 |
| Safe Auto-Handling Rate | — | **0.255** | **0.255** | **0.255** |
| False Auto-handle (should-escalate) | — | **0.243** | 0.270 | 0.322 |
| Escalation F1 | — | **0.767** | 0.760 | 0.726 |
| Reply correctness | — | 4.36 | **4.74** | **4.78** |
| Reply groundedness | — | 4.33 | **4.74** | **4.77** |
| Unsupported-claim rate | — | **0.015** | 0.020 | 0.020 |
| Safety suite | — | **6/6**† | **6/6**† | **6/6**† |
| Mean latency (s) | — | — | **1.97** | 6.98 |

† Suite scores after assertion-aware harness fixes on the **same stored replies**: DeepSeek `fake_policy` (safe negation of `always guaranteed`); GPT `account_specific` (user redirect containing `your last transaction`). Not model-behavior improvements. Default remains GPT-4o-mini; SAH stays **0.255**.

**Selection:** keep **GPT-4o-mini** as the primary ResolveFlow model. Safe auto-handle is tied (0.255), but GPT has lower false auto-handle and higher escalation F1. DeepSeek non-thinking wins judged reply quality; thinking mode does **not** improve safe automation and is ~3.5× slower with worse FAH.

Artifacts: `artifacts/final/model_comparison.md`, `artifacts/final/deepseek_v41_flash_{nonthinking,thinking}/`.

### Run DeepSeek experiments

```bash
python scripts/run_experiment.py --config configs/experiments/deepseek_v41_flash_nonthinking.yaml --force
python scripts/run_experiment.py --config configs/experiments/deepseek_v41_flash_thinking.yaml --force
python scripts/compare_llm_models.py
```

Artifacts are isolated and do **not** overwrite GPT-4o-mini results.

## Architecture

```text
Customer → Intent (gpt-4o-mini) → Retrieve (k=3) → Draft (gpt-4o-mini) → Safety → Escalation → Auto / Human
```

Code: `src/resolveflow/agent/`, retrieval in `src/resolveflow/retrieval/`, eval in `evaluation/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,discovery,labeling,llm]"

# Put your key in .env (gitignored)
# OPENAI_API_KEY=sk-...
# RESOLVEFLOW_LLM_MODEL=gpt-4o-mini

# Works from a fresh clone (committed golden + artifacts/final)
pytest
python scripts/check_golden_set.py
python scripts/check_leakage.py --final
python scripts/check_taxonomy.py
cat artifacts/final/headline.json   # Safe Auto-Handling Rate = 0.255

# Offline agent / retrieve / safety suite need local data/processed/*
# (gitignored). Build once from twcs.csv — see Data below — then:
python scripts/run_agent.py --offline --text "My package is late and tracking hasn't moved"
python scripts/retrieve.py --text "My package is late and tracking hasn't moved"
python scripts/evaluate_safety.py --mode offline

# Live agent (uses .env → gpt-4o-mini; also needs data/processed)
python scripts/run_agent.py --text "My package is late and tracking hasn't moved"
```

### What a fresh clone can verify without rebuilding data

| Check | Needs |
| --- | --- |
| `pytest`, golden/leakage/taxonomy | Committed repo only |
| Headline / comparison tables | `artifacts/final/` (committed) |
| Offline agent, retrieve, offline safety suite | Local `data/processed/*` (+ MiniLM download) |
| Full OpenAI eval / GPT safety suite | API key + `data/processed/*` |

## Evaluation

**First OpenAI run** (API cost; writes `artifacts/final/`):

```bash
python scripts/evaluate_agent.py --mode openai
python scripts/evaluate_safety.py --mode openai
python -m evaluation.run_all --config configs/default.yaml --mode openai
```

**Cached reproduction:** re-run `evaluation.run_all --mode openai` to reuse `artifacts/evaluation/agent_predictions.jsonl` and `artifacts/final/judge_cache.json` unless you pass `--force-agent`.

Related:

```bash
python scripts/evaluate_baselines.py
python scripts/evaluate_retrieval.py
streamlit run scripts/rate_replies.py   # human reply ratings
```

Artifacts: `artifacts/final/` (metrics, comparisons, manifest), `artifacts/figures/escalation_tradeoff.png`.

### Main comparison

| System | Intent Macro-F1 | Escalation F1 | Auto-Handle | FAH | Reply Correctness | Groundedness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.012 | 0.730 | 0.000 | 0.000 | 3.24 | 2.38 |
| TF-IDF + LR | 0.683 | 0.592 | 0.375 | 0.383 | — | — |
| Nearest Case | — | — | — | — | 3.46 | 3.42 |
| ResolveFlow (gpt-4o-mini) | 0.669 | 0.767 | 0.440 | 0.243 | 4.36 | 4.33 |

## Failure Analysis

Top issues: false auto-handle, refund request/status confusion, delivery vs missing-package confusion, thin/ambiguous tweets, occasional unsupported claims (~1.5%). Safety suite is **6/6** after fixing harness false positives (`fake_policy` negations; GPT `account_specific` redirects)—not by changing model defaults. Details: [`report/failure_analysis.md`](report/failure_analysis.md).

## Limitations

- Golden n=200; stratified Twitter sample; one brand (AmazonHelp).
- LLM intent Macro-F1 does **not** beat TF-IDF on this set (CI overlaps).
- LLM-as-judge calibrated lightly (n=40 solo); correctness Spearman ≈ 0.30.
- Safety suite: **6/6** after assertion-aware harness regrades (`fake_policy`, `account_specific`). No live CSAT or account APIs.
- Retrieval ablations: k=0 blocks auto-handle; k=1/3/5 similar automation.

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
python scripts/profile_data.py --write-interim
python scripts/build_cases.py
python scripts/build_index.py
```

Never commit `.env` or API keys.
