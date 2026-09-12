# Citations / Attribution

External datasets, models, and libraries that materially contribute to ResolveFlow.
This is attribution, not a literature review.

## Dataset

| Source | What was used | Where |
| --- | --- | --- |
| **Customer Support on Twitter (TWCS)** — Kaggle dataset [`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | Historical public Twitter customer-support conversations; brand slice **AmazonHelp** | Raw CSV expected at `data/raw/twcs.csv` (not committed; ~493 MB). Ingest: `src/resolveflow/data/ingest.py`, `scripts/profile_data.py`, `scripts/build_cases.py` |
| Hugging Face mirror [`SunidhiSriram/twcs`](https://huggingface.co/datasets/SunidhiSriram/twcs) (`twcs.csv`) | Optional download path when Kaggle credentials are unavailable | `scripts/download_dataset.py` (`HF_URL`) |

Schema/row-count expectations are documented in `report/methodology.md`. Prefer Kaggle as the canonical source when credentials are available; re-verify file identity against Kaggle if using the mirror.

## Embedding / retrieval

| Source | What was used | Where |
| --- | --- | --- |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Sentence Transformers) | Dense embeddings for historical-case retrieval (default `rerank_mode: none`) | `configs/default.yaml` (`embedding_model`), `src/resolveflow/retrieval/`, `scripts/build_index.py`, intent discovery embeddings in `scripts/discover_intents.py` |

## LLM providers / models

| Source | What was used | Where |
| --- | --- | --- |
| **OpenAI GPT-4o-mini** | Default classifier, responder, and LLM-as-judge (`judge_v1`) | `configs/default.yaml`, `src/resolveflow/llm/`, evaluation under `artifacts/evaluation/agent_predictions.jsonl` |
| **DeepSeek API** | Controlled model-substitution benchmark (not the default production config) | See DeepSeek note below |

### DeepSeek model identifier (evidence-backed)

Recorded in experiment configs and manifests (not renamed for cosmetics):

| Field | Value |
| --- | --- |
| API model string sent to provider | `deepseek-flash` |
| Base URL | `https://api.deepseek.com` |
| Documented product/version label in repo | `DeepSeek-V4.1-Flash` (`model_version` in YAML / `DeepSeekProvider.MODEL_VERSION`) |
| Thinking control | `extra_body={"thinking":{"type":"enabled"|"disabled"}}`; thinking runs also set `reasoning_effort=high` |
| Artifact dirs | `artifacts/final/deepseek_v41_flash_{nonthinking,thinking}/` |
| Artifact timestamps (filesystem) | Predictions/judge caches dated **2026-09-10** (local run); no provider-side immutable snapshot ID is stored in responses |

**Provider docs (as of this write-up):** DeepSeek lists API id `deepseek-flash` with model version **DeepSeek-V4.1-Flash** on [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing). Legacy aliases such as `deepseek-v4-flash` are documented as temporary compatibility routes to V4.1-Flash; this repo used the canonical id `deepseek-flash`, not those aliases.

**Limits of evidence:** Per-example prediction `meta` records `provider: deepseek` but does **not** store the raw API `model` echo or a dated immutable snapshot id. Token usage sums in DeepSeek manifests are `null`. The version string `DeepSeek-V4.1-Flash` is the repo’s configured label aligned with provider docs for `deepseek-flash`; it is not independently recovered from a frozen response header in the artifacts.

Code: `src/resolveflow/llm/deepseek.py`, `configs/experiments/deepseek_v41_flash_*.yaml`, `scripts/run_experiment.py`, `scripts/compare_llm_models.py`.

## Evaluation methodology (assignment-driven, not borrowed code)

| Item | Notes |
| --- | --- |
| LLM-as-judge rubric (`judge_v1`) | Implemented for this take-home; human calibration sample n=40 (solo). Agreement artifact: `artifacts/final/judge_human_agreement.json` |
| Bootstrap CIs | Percentile bootstrap consistent with Macro-F1 machinery in `evaluation/run_all.py`; headline CIs via `scripts/compute_headline_cis.py` |
| Metrics | Intent Macro-F1 / escalation P/R/F1 via scikit-learn (`src/resolveflow/metrics.py`) |

No third-party prompt pack or evaluation harness was copied into this repository as a dependency. Patterns (risk-aware escalation, dense retrieval, LLM-as-judge) are standard industry practice, implemented in-repo.

## Material libraries

| Library | Role |
| --- | --- |
| `scikit-learn` | TF-IDF+LR baseline, classification/escalation metrics |
| `numpy` / `pandas` | Numerics and data handling |
| `sentence-transformers` | MiniLM embeddings (optional extra `discovery`) |
| `openai` | OpenAI + DeepSeek OpenAI-compatible clients (optional extra `llm`) |
| `streamlit` | Human rating UI (`scripts/rate_replies.py`) |
| `pytest` | Regression / integrity tests |

See `pyproject.toml` for versions/extras. License obligations follow each package’s upstream license when redistributing.
