# Golden Set Methodology

## Brand

AmazonHelp (selected in Phase 1).

## How were examples selected?

1. Build all AmazonHelp customer messages (~203k).
2. Prefer English-dominant text for the MVP taxonomy (documented limitation).
3. Draw a **stratified candidate pool (220)** mixing:
   - keyword pools aligned to candidate intents (oversample rarer patterns)
   - hard/short/ambiguous messages
   - longer threads (≥5 messages)
   - random remainder
4. Cap at most 2 messages per conversation in the candidate pool; final set keeps
   **one conversation per example** after top-ups.
5. Top-up rare intents (`refund_status`, `technical_issue`, etc.) from the full
   brand corpus with conversation exclusion against the emerging golden set.
6. Trim to **200** labelled examples.

This is **not** a pure random production sample.

## How many examples?

**200** (within the 150–250 target).

## Was sampling stratified?

Yes — by keyword intent pools, difficulty/hard cases, thread length, plus random fill.

## Were rare intents oversampled?

Yes. Natural frequency would under-represent `refund_status` / some account & tech
issues. Oversampling makes per-class evaluation possible. **Aggregate accuracy is
therefore not an estimate of production traffic mix.** Prefer macro-F1 alongside
accuracy in later phases.

## Were hard cases deliberately included?

Yes (~hard stratum + short/ambiguous messages). Difficulty labels: easy / medium / hard.

## Were conversations kept intact?

Each golden row is a single customer message with optional prior-turn `context`.
Entire conversations are **excluded** from development/retrieval via conversation_id
(and text) leakage checks.

## How were labels assigned?

Solo annotator following `data/golden/ANNOTATION_GUIDE.md`.

Workflow:

```text
candidates → rule-assisted draft (resolveflow.annotate)
          → manual overrides from reading review file
          → rare-intent top-up with guideline checks
          → validate → freeze + SHA-256
```

This is **not** "LLM labels 200 rows unsupervised." Rule assist encodes the written
guide; ambiguous cases were reviewed. No second annotator was available in this
environment.

## Was a second annotator used?

**No.** Inter-annotator agreement was not computed. This is a limitation.

## Was the set frozen?

Yes — `python scripts/freeze_golden_set.py` writes `golden_set_manifest.json` with
`status: FROZEN` and SHA-256.

## Development / leakage split

`python scripts/check_leakage.py --build-dev` builds
`data/interim/development_messages.parquet` excluding golden conversation IDs,
message IDs, exact texts, and normalized texts.

## Limitations / selection bias

> The golden set was intentionally stratified rather than sampled according to
> natural production frequency. Consequently, aggregate metrics should not be
> interpreted as estimates of real-world production performance.

Additional limits: English preference; solo annotation; Twitter public channel bias
(DM redirects common); brand replies not used as intent labels.
