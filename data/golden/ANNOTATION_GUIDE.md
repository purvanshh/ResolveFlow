# ResolveFlow Annotation Guide

## Goal

Assign the customer's **primary support intent** and decide whether the case
should be handled automatically or escalated. Labels are for the frozen golden
evaluation set only.

Brand in scope: **AmazonHelp**.

## General Rules

1. Label the customer's actual request in `input_text`, using `context` when needed.
2. Do **not** infer facts that are not present in the message or context.
3. Choose **one primary intent**.
4. Do **not** let the brand's reply dictate the intent.
5. Use `other_unclear` when evidence is insufficient.
6. Escalation is about **operational risk / information needs**, not anger alone.
7. Do not look at model predictions while labeling.

## Multi-intent policy

If a message contains multiple issues (e.g. "charged twice and I want a refund"):

> Assign the intent for the **primary actionable support problem**.

Record secondary intents in `annotation_notes` if useful.

Example: duplicate charge + refund ask → usually `payment_billing` when the charge
anomaly is the root issue; use `refund_request` when the only ask is to start a refund
without a billing anomaly.

## Intent Definitions

### delivery_delay

Late / delayed shipment vs promised date; package still expected.

### package_missing_or_misdelivered

Marked delivered but missing, wrong address, stolen, courier left incorrectly.

### order_quality_issue

Received but damaged, defective, wrong, fake, or incomplete.

### cancellation_request

Customer wants the order cancelled (or refuses upcoming delivery as cancel).

### return_request

Wants to start a return / label / pickup for something received.

### refund_request

Wants to **initiate** a refund / get money back (refund not already underway).

### refund_status

Refund already requested/approved; asking where it is / why delayed.

### payment_billing

Charges, failed payments, Amazon Pay, gift cards, duplicate charges, promo/billing.

### prime_membership

Prime subscription itself (signup/renew/cancel/benefits), not mere Prime shipping delay.

### account_access

Login failures, locked/hacked/closed account, credential problems.

### technical_issue

App/site bugs, crashes, unavailable pages, checkout UI failures.

### other_unclear

Insufficient evidence: acknowledgements, URL-only, yes/no without recoverable issue,
non-actionable chatter, or equally plausible multiple intents.

## Escalation

Values: `auto_handle` | `escalate` | `uncertain`

### Auto-handle

Intent clear; a useful grounded public reply is possible without account-secret
actions; low financial/security risk.

### Escalate

Needs account verification, financial action, security/fraud, missing-package
investigation, or intent/evidence is insufficient for a safe auto reply.

### Uncertain

Use sparingly. If frequent, the policy is too vague.

If `escalate`, **escalation_reason is required**.

## Difficulty

- **easy** — clear request, little ambiguity
- **medium** — some context needed
- **hard** — ambiguous, noisy, multi-intent, thin context, or safety-sensitive

## Examples

| Text | Intent | Escalation |
| --- | --- | --- |
| "Prime next-day still not here, now says Friday" | delivery_delay | auto_handle |
| "Says delivered but nothing arrived" | package_missing_or_misdelivered | escalate |
| "I want a refund" | refund_request | escalate |
| "Refund approved 7 days ago — where is it?" | refund_status | escalate |
| "Still waiting." (no ctx) | other_unclear | escalate |

## Annotator note (this project)

Golden labels in v1 were produced by a **solo annotator** following this guide,
with a rule-assisted draft (`resolveflow.annotate`) plus manual overrides from
reading candidates. This is **not** a bulk LLM auto-label pipeline, and labels
must not be silently edited after freeze to improve model scores.
