# Agent Architecture — Phase 4

## Runtime note (this environment)

No `OPENAI_API_KEY` was available. Reported Phase-4 numbers use the **offline agent**:

| Component | Implementation |
| --- | --- |
| Intent | Phase-3 TF-IDF + Logistic Regression |
| Retrieval | MiniLM index (dev-only historical cases) |
| Reply | Grounded template drafter (no invented refunds/timelines/actions) |
| Safety | Deterministic validators |
| Escalation | Risk-aware policy |

OpenAI path is implemented (`OpenAIProvider`, `--mode openai` / `auto` with key). Mock LLM is for unit tests only — not used as the reported evaluation classifier when offline mode selects TF-IDF.

## Architecture

```text
message → context → classify → retrieve → (optional draft) → safety → policy → decision
```

Modules: `src/resolveflow/agent/{classifier,responder,safety,policy,pipeline,factory}.py`

## Intent classification

- LLM prompt: `prompts/classifier.py` (`classifier_v1`) with taxonomy include/exclude.
- Invalid intents retry once, then `other_unclear` + escalate.
- Thin messages (≤20 chars, little context) force `other_unclear`.
- Offline eval uses TF-IDF (same weights as Phase 3 baseline).

## Retrieval

- Mode `auto`: prefer intent-filtered hits; fall back to global if too few.
- Threshold + `is_sufficient_evidence()` gate automation.
- Golden conversations never enter the index (Phase 3 leakage tests still apply).

## Response generation

- LLM prompt: `prompts/responder.py` (`responder_v1`) with untrusted delimiters for customer/evidence.
- Offline: cautious templates keyed by intent; may mention private channel if historical replies do.
- Never claims completed refunds/cancellations or dollar/timeline amounts without evidence support.

## Safety

Deterministic flags: monetary, timeline, completed-action, policy, account-visibility, injection, length, missing evidence.

## Escalation

Auto-handle only if:

```text
known non-high-risk intent
AND confidence ≥ threshold
AND sufficient evidence
AND safety PASS
AND no explicit human request
```

High-risk intents (configured): payment/refund/account/missing-package.

Conservative default: escalate on failure.

## Ablations (offline, golden n=200)

See `artifacts/evaluation/retrieval_ablation.json`. Intent Macro-F1 is unchanged across K (classifier independent of K). Automation rate rises only when retrieval evidence is present (K≥1). Unsupported-claim rate stayed **0.0** for drafted replies.

## Limitations

- Offline numbers are **not** a hosted-LLM agent score; swap in OpenAI for that comparison.
- False-auto-handle remains non-trivial: some gold “escalate” cases are low-risk intents with good evidence (policy disagreement / conservatism mismatch).
- Template replies are safer than fluent LLM prose but less natural.
- TF-IDF still confuses `refund_status` vs `refund_request` and under-recalls `technical_issue`.
