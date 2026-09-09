# Retrieval Analysis — Examples

Intent Recall@K is a proxy. These examples show how retrieval behaves on real AmazonHelp-style queries.

## 1. Good retrieval (excellent)

**Query**

> I was charged twice for the same transaction

**Top hit** (similarity ≈ 0.73, intent=`payment_billing`)

> Customer: "…yes it said I was charged twice."
> Brand: asks customer to contact phone/chat for investigation.

Why good: same operational problem (duplicate charge) and a historically used handling pattern (private channel investigation). Useful grounding for Phase 4 — not a license to invent a refund timeline.

## 2. Mediocre / topic-only retrieval

**Query**

> Says delivered but nothing arrived at my door

**Top hit** (similarity ≈ 0.66, intent=`other_unclear` silver)

> Customer: order "supposed to be delivered today" but door inaccessible…
> Brand: asks which carrier.

Same delivery neighborhood, different issue (access/failed attempt framing vs confirmed "delivered but missing"). Rank-2/3 are closer (`package_missing_or_misdelivered`). Shows why top-1-only can be brittle and why intent match ≠ resolution match.

## 3. Weak / failure-ish retrieval

**Query**

> Still waiting.

Without context, nearest neighbors tend to be other vague chase messages or generic delays. Similarity can still look "okay" because short tweets cluster, but **evidence is not actionable**. Correct system behavior: treat as insufficient evidence → escalate (`other_unclear`), not copy a DM template.

## Misleading retrieval risk

Nearest-neighbor reply baseline may copy:

> "Please contact us via phone or chat…"

into a new case where the right move differs (e.g. refund already approved). Phase 4 must **synthesize grounded answers** and refuse unsupported claims — not paste historical text blindly.

## Categories checklist

| Category | Observed? |
| --- | --- |
| Good retrieval | Yes (duplicate charge) |
| Topic-only | Yes (delivery neighborhood) |
| Wrong intent in top-1 | Yes (silver `other_unclear` on missing-package query) |
| Weak / no useful evidence | Yes (ultra-short follow-ups) |
| Misleading resolution if copied | Yes (generic DM redirects) |
