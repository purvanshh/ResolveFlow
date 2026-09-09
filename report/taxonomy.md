# Taxonomy — AmazonHelp

## How this taxonomy was built

1. Reconstructed AmazonHelp conversations (Phase 1).
2. Built ~203k customer messages; sampled **4,000** for discovery
   (`data/interim/intent_discovery_sample.parquet`).
3. Embedded with `all-MiniLM-L6-v2`, clustered with KMeans **k ∈ {8,10,12,15}**.
4. Inspected representative examples + TF-IDF keywords (`data/interim/clusters/`).
5. **Did not** adopt clusters as labels. Clusters mixed languages, acknowledgements,
   and operationally different asks (e.g. refund initiate vs refund status).
6. Final **12 intents** chosen for operational usefulness on Amazon support traffic.

## Intent list

1. `delivery_delay`
2. `package_missing_or_misdelivered`
3. `order_quality_issue`
4. `cancellation_request`
5. `return_request`
6. `refund_request`
7. `refund_status`
8. `payment_billing`
9. `prime_membership`
10. `account_access`
11. `technical_issue`
12. `other_unclear`

Full include/exclude/confusing_with rules: `configs/intents.yaml`.

## Multi-intent policy

One **primary** actionable intent per example. Secondary notes allowed in
`annotation_notes`.

## Hardest boundaries

### refund_request vs refund_status

**Boundary**

- `refund_request`: customer asks to start a refund / get money back; no established
  refund process in message/context.
- `refund_status`: refund already requested/approved/processed; customer asks where
  it is or why it has not landed.

**Ambiguous example**

> "I've been waiting for my money."

Why ambiguous: does not establish whether a refund was initiated.

**Policy:** `other_unclear` unless context shows a prior refund commitment.

### delivery_delay vs package_missing_or_misdelivered

**Boundary**

- Delay: still expected; complaint is timing.
- Missing/misdelivered: claimed delivered / wrong place / not in possession.

**Ambiguous example**

> "Where is my package?"

**Policy:** use context/tracking language; if only "late", prefer `delivery_delay`;
if "says delivered", prefer `package_missing_or_misdelivered`; if neither, `other_unclear`.

### refund_request vs payment_billing

Duplicate charge often co-occurs with "refund me".

**Policy:** if the anomaly is the charge itself → `payment_billing`; if the only ask
is to refund a normal order/return → `refund_request`.

### prime_membership vs delivery_delay

Mentioning Prime does not make it a membership intent.

**Policy:** Prime subscription/benefits/cancel → `prime_membership`; late Prime
shipping → `delivery_delay`.

## Limitations

- English-preferring golden sampling; FR/ES/DE/JP traffic exists in the raw brand data.
- No dedicated "seller complaint" or "pricing error" intents — folded into nearest
  operational class or `other_unclear`.
- Clusters showed many acknowledgement-only tweets; `other_unclear` is intentional.
