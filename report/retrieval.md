# Retrieval System — Phase 3

## Corpus

| Field | Value |
| --- | --- |
| Brand | AmazonHelp |
| Historical cases | **15,000** |
| Source | Development customer messages with a brand reply |
| Golden exclusion | Conversation IDs, message IDs, and exact texts excluded |
| Case unit | One customer turn + following brand response (+ optional prior context) |
| Resolution summary | Deterministic paraphrase of the **actual** brand reply (no invented outcomes) |

Built via `python scripts/build_cases.py` → `data/processed/historical_cases.parquet`.

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2` (384-d), L2-normalized cosine via dot product.

Why: small, fast, widely used, fits a ~15-minute runnable pipeline.

## Index

Brute-force matrix multiply over cached `embeddings.npy` (15k × 384). No external ANN service.

Fingerprint covers: case checksum + model + preprocessing version + count. Rebuild only when fingerprint changes (`scripts/build_index.py`).

## Retrieval metrics (golden set, n=200)

Intent Recall@K proxy — ≥1 of top-K retrieved cases has the **same silver/rule intent label as the golden intent**:

| Metric | Value |
| ---: | ---: |
| Recall@1 | **0.310** |
| Recall@3 | **0.565** |
| Recall@5 | **0.670** |
| Mean top-1 cosine | **0.756** |

### Limitation

> Intent-level retrieval relevance is **not** equivalent to true support-resolution relevance.
> Same intent can still imply different operational next steps (e.g. refund request vs status).

Historical case intents are **silver labels** (Phase-2 annotation rules on development text), while golden intents are human labels — another source of mismatch.

## Manual relevance sample

Heuristic 0–2 scores on 40 queries (intent match + similarity bands) are stored in
`artifacts/metrics/retrieval.json` as a diagnostic proxy. They are **not** a full
double-blind human study.

## Failure modes

See `report/retrieval_analysis.md` for concrete examples:

- Good retrieval
- Topic-only retrieval
- Wrong intent
- Weak / below-threshold
- Misleading resolution copy risk

## Phase 4 prep

`format_evidence()` produces compact CASE blocks for top-k ∈ {0,1,3,5} ablations.
Nearest-neighbor reply baseline copies `brand_response` from top-1 (intentionally naive).
