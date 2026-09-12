# ResolveFlow

A grounded AI support agent for **AmazonHelp**.

Classify intent → retrieve historical cases → draft a grounded reply → safety check → escalate or auto-handle.

## Results

Frozen golden set **n=200** (checksum `e974255a…bb4ef36`).  
**Fingerprint:** `gpt-4o-mini` classifier + responder + MiniLM retrieval (k=3, `rerank_mode=none`) + risk-aware escalation (`order_quality_issue` high-risk + message-level risk cues) + `gpt-4o-mini` judge.

**Safe Auto-Handling Rate** = auto-handled **and** correct intent **and** no safety flags **and** gold also auto_handle (**51/200 = 0.255**).  
**Auto-handle rate (0.320)** is only “policy did not escalate”—not safe automation.

| Metric | Point estimate | 95% bootstrap CI |
| --- | ---: | ---: |
| **Safe Auto-Handling Rate** | **0.255** (51/200) | [0.195, 0.310] |
| Auto-handle rate | 0.320 (64/200) | [0.262, 0.385] |
| False auto-handle (among should-escalate) | **0.043** (5/115) | [0.009, 0.089] |
| Escalation F1 | **0.876** | [0.831, 0.916] |
| Intent Macro-F1 (ResolveFlow / LLM) | 0.669 | [0.598, 0.727] |
| TF-IDF Macro-F1 (baseline) | **0.683** | — |
| Safety suite | **6/6 PASS** | — |

CIs from frozen GPT predictions (`scripts/compute_headline_cis.py` → `artifacts/final/headline_confidence_intervals.json`). Point estimates unchanged.

**Safety regression (separate from golden headline):** the original 6 escalate-only cases remain a smoke subset (**6/6**). They were expanded into a **mixed-outcome** adversarial suite (n=36; 19 must-escalate / 17 safe-auto) so always-escalate cannot score as perfect. Offline run: escalation recall **1.0**, FAH **0.0**, false-escalation **2/17**, balanced score **0.94** vs always-escalate **0.50**. See `artifacts/final/safety_suite_mixed.json`. Not a production estimate; not merged into SAH/FAH.

Escalation path (same frozen predictions): FAH **0.243 → 0.096 → 0.043** with SAH held at **0.255**. Hybrid GPT+TF-IDF was tested and **not** adopted. **Caveat:** policy/cues were selected on this golden set (labels/examples frozen)—FAH/SAH may be optimistic vs a held-out calibration split.

On this set, **TF-IDF edges the LLM on intent Macro-F1**. Soft reply-quality comparisons use an LLM judge that is **weakly human-validated** on correctness/helpfulness (see report §6).

Full story: [`report/report.md`](report/report.md). Headline limits: report §10. Sources: [`CITATIONS.md`](CITATIONS.md).

## LLM Model Comparison

Controlled substitution on the same frozen golden set (n=200), retrieval (k=3), safety, escalation, and **gpt-4o-mini judge**.

**DeepSeek:** API model id **`deepseek-flash`**, base URL `https://api.deepseek.com`, repo version label **`DeepSeek-V4.1-Flash`** (aligned with DeepSeek’s current pricing docs for that id). Details: [`CITATIONS.md`](CITATIONS.md).

| Metric | TF-IDF | GPT-4o-mini | DeepSeek Non-thinking | DeepSeek Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | **0.683** | 0.669 | 0.653 | 0.634 |
| Auto-handle rate | — | 0.320 | 0.355 | 0.365 |
| Safe Auto-Handling Rate | — | **0.255** | **0.255** | **0.255** |
| False Auto-handle (should-escalate) | — | **0.043** | 0.087 | 0.096 |
| Escalation F1 | — | **0.876** | 0.861 | 0.860 |
| Reply correctness (LLM judge) | — | 4.36 | 4.74 | 4.78 |
| Reply groundedness (LLM judge) | — | 4.33 | 4.74 | 4.77 |
| Unsupported-claim rate | — | **0.015** | 0.020 | 0.020 |
| Safety suite | — | **6/6**† | **6/6**† | **6/6**† |
| Mean latency (s) | — | — | **1.97** | 6.98 |

† Safety harness regrades on stored replies. Escalation includes message risk cues; intents/replies frozen.

**Selection:** keep **GPT-4o-mini** (lowest FAH, highest escalation F1 at tied SAH). On the LLM judge’s scoring, DeepSeek received higher soft reply-quality scores; human agreement for those dimensions was weak (helpfulness Spearman ≈ −0.19, overall κ ≈ −0.13), so treat that as **directional**, not a validated quality win. Thinking mode does **not** improve safe automation and is ~3.5× slower.

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
Customer → Intent (gpt-4o-mini / classifier_v1)
        → Retrieve (MiniLM, k=3, rerank_mode=none)
        → Draft (gpt-4o-mini)
        → Safety validator
        → Escalation (high-risk intents + message risk cues)
        → Auto-handle / Human
```

Code: `src/resolveflow/agent/`, retrieval in `src/resolveflow/retrieval/`, eval in `evaluation/`.

## Quickstart — artifact verification vs recomputation

### Artifact verification (no API; under ~15 minutes)

The repository **commits frozen prediction artifacts** used for the reported headline metrics. A fresh clone can **verify the exact calculation** without an API call. `artifacts/final/headline.json` is a **committed summary of those frozen predictions**, not an independently recomputed live eval.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,discovery,labeling,llm]"

# Optional: .env for live calls only (gitignored)
# OPENAI_API_KEY=sk-...
# RESOLVEFLOW_LLM_MODEL=gpt-4o-mini

pytest
python scripts/check_golden_set.py
python scripts/check_leakage.py --final
python scripts/check_taxonomy.py
python scripts/compute_headline_cis.py    # CIs from frozen predictions (no API)
cat artifacts/final/headline.json         # SAH = 0.255 (committed summary of frozen preds)
# Mixed safety suite needs local data/processed (offline agent); see below
```

### Needs local processed data (no API)

Offline agent / retrieve / offline safety paths need `data/processed/*` (gitignored; MiniLM download). Build once from `twcs.csv` — see Data. The raw ~493 MB CSV is **not** committed.

```bash
python scripts/run_agent.py --offline --text "My package is late and tracking hasn't moved"
python scripts/retrieve.py --text "My package is late and tracking hasn't moved"
python scripts/evaluate_safety.py --mode offline   # mixed-outcome suite + always-escalate baseline
```

### True recomputation (API + processed data)

Re-running the model from scratch requires the processed dataset **and** an OpenAI API key (and matching model config). This is **not** the under-15-minute path.

```bash
python scripts/evaluate_agent.py --mode openai
python scripts/evaluate_safety.py --mode openai
python -m evaluation.run_all --config configs/default.yaml --mode openai
```

**Cached reproduction:** `evaluation.run_all --mode openai` reuses `artifacts/evaluation/agent_predictions.jsonl` and `artifacts/final/judge_cache.json` unless `--force-agent`.

| Check | Needs |
| --- | --- |
| `pytest`, golden/leakage/taxonomy, headline CI from preds | Committed repo only |
| Inspect `artifacts/final/*` tables | Committed artifacts |
| Offline agent / retrieve / offline safety | Local `data/processed/*` |
| Fresh GPT eval / live GPT safety suite | API key + `data/processed/*` |

## Evaluation extras

```bash
python scripts/evaluate_baselines.py
python scripts/evaluate_retrieval.py
streamlit run scripts/rate_replies.py   # human reply ratings
```

### Main comparison (soft reply columns = LLM-judge directional)

| System | Intent Macro-F1 | Escalation F1 | Auto-Handle | FAH | Reply Correctness† | Groundedness† |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.012 | 0.730 | 0.000 | 0.000 | 3.24 | 2.38 |
| TF-IDF + LR | 0.683 | 0.592 | 0.375 | 0.383 | — | — |
| Nearest Case | — | — | — | — | 3.46 | 3.42 |
| ResolveFlow (gpt-4o-mini) | 0.669 | 0.876 | 0.320 | 0.043 | 4.36 | 4.33 |

† LLM-judge means; human agreement weak on soft quality (correctness κ≈0.15; helpfulness κ≈−0.05). Hallucination/safety agreement is stronger (κ≈0.79).

## Failure Analysis

Top issues: residual golden FAH (**5/115** boundary cases), refund status/request confusion, delivery vs missing-package confusion, thin/ambiguous tweets. Legacy safety smoke **6/6**; mixed-outcome safety suite documents **2 false escalations** on borderline benign FAQs (no production change). Details: [`report/failure_analysis.md`](report/failure_analysis.md).

## Limitations

- Golden n=200; stratified Twitter sample; one brand (AmazonHelp).
- Escalation policy / message-risk cues selected on the same frozen eval set → possible optimistic FAH/SAH.
- LLM intent Macro-F1 does **not** beat TF-IDF on this set.
- LLM-as-judge: strong hallucination/safety agreement (Spearman ≈ 0.679, weighted κ ≈ 0.792); weak/negative on helpfulness and overall — soft quality is directional only.
- Legacy safety smoke (6 escalate-only cases) is insufficient alone; mixed suite (n=36) is a regression harness, not production coverage. No live CSAT or account APIs.

## Docs

| Doc | Path |
| --- | --- |
| Final report | `report/report.md` |
| Decision log | `report/decision_log.md` |
| Failure analysis | `report/failure_analysis.md` |
| Citations / attribution | `CITATIONS.md` |
| Interview notes | `report/interview_notes.md` |
| Taxonomy | `report/taxonomy.md` |
| Methodology | `report/methodology.md` |

## Attribution

External datasets, models, and libraries: [`CITATIONS.md`](CITATIONS.md).

## Data

Place `twcs.csv` under `data/raw/` (gitignored). Processed embeddings/cases are local artifacts—not committed.

```bash
python scripts/profile_data.py --write-interim
python scripts/build_cases.py
python scripts/build_index.py
```

Never commit `.env` or API keys.
