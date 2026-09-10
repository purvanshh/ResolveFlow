# ResolveFlow — AI Support Agent for AmazonHelp

## 1. Executive Summary

ResolveFlow is a grounded customer-support agent prototype for **AmazonHelp** tweets. It classifies intent (12-class brand taxonomy), retrieves historical cases, drafts a conservative reply, runs safety checks, and applies a risk-aware escalation policy.

**Headline metric: Safe Auto-Handling Rate = 0.165**  
(auto-handled ∧ correct intent ∧ no safety flags ∧ gold also auto-handle; n=200 frozen golden set)

Supporting numbers (offline eval: TF-IDF classifier + grounded template responder + MiniLM retrieval):

| Metric | Value |
| --- | ---: |
| Intent Macro-F1 | 0.683 (95% bootstrap CI [0.618, 0.745]) |
| Auto-handle rate | 0.260 |
| False auto-handle rate (among should-escalate) | 0.165 |
| Escalation F1 | 0.730 |
| Retrieval Recall@3 | 0.565 |
| Safety suite | 6/6 PASS; unsupported-claim rate 0.0 |

The intent head matches TF-IDF (same classifier offline). The agent’s contribution is safer automation than confidence-only escalation (FAH 0.165 vs 0.383) with zero fabricated policy/action claims on the golden set.

---

## 2. Problem Framing

Good is **useful automation that refuses unsupported claims and escalates when account-specific or high-risk**. Maximizing auto-handle rate alone is the wrong objective. False auto-handling is treated as the primary safety failure.

---

## 3. Scope

**Built:** intent classification, historical retrieval, grounded reply drafting, escalation policy, safety gates, baselines, frozen golden evaluation, LLM/heuristic judge + human calibration scaffold, failure analysis.

**Not built:** live account access, refunds/CRM actions, production deployment, multi-brand support, real-time Twitter integration.

---

## 4. Data & Methodology

- Dataset: Customer Support on Twitter (`twcs.csv`); brand **AmazonHelp**.
- Taxonomy: 12 intents (`configs/intents.yaml`), including `other_unclear` for abstention.
- Golden set: **200** frozen examples; checksum `e974255a…bb4ef36`; solo + rule-assisted annotation (not bulk LLM labels).
- Splits: conversation-level; golden excluded from development messages and retrieval corpus.
- Final leakage audit: conversation / exact / normalized / retrieval overlap = **0** (`STATUS: PASS`).
- Final eval mode: **offline** (no API key). OpenAI classifier/responder paths exist but are not the reported fingerprint.

---

## 5. Architecture

```text
Customer message
   ↓
Intent (TF-IDF offline / LLM optional)
   ↓
Retrieve historical cases (MiniLM, k=3)
   ↓
Draft grounded response (template / LLM)
   ↓
Safety validation
   ↓
Risk-aware escalation policy
   ↓
Auto-handle  /  Human escalate
```

---

## 6. Evaluation

- Frozen golden set + automated intent/escalation metrics + bootstrap CI.
- Baselines: majority, TF-IDF+LR, always-escalate, confidence threshold, nearest historical reply, generic reply.
- Reply judge (`judge_v1`): structured 1–5 rubric; cached; heuristic offline (OpenAI path available).
- Human calibration: 40 examples, solo annotator; Streamlit UI `scripts/rate_replies.py`.
- Ablations: retrieval k∈{0,1,3,5}; no-retrieval blocks auto-handle; escalation policy comparison.
- Safety suite: unsupported refunds, injection, fake policy, account-specific, ambiguous, false action.

---

## 7. Results

### Intent

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Majority | 0.075 | 0.012 | 0.010 |
| TF-IDF + LR | 0.695 | 0.683 | 0.676 |
| ResolveFlow (offline) | 0.695 | 0.683 | 0.676 |

### Difficulty (accuracy / macro-F1 over intents present)

| Difficulty | n | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Easy | 58 | 0.879 | 0.750 |
| Medium | 128 | 0.641 | 0.535 |
| Hard | 14 | 0.429 | 0.254 |

### Escalation (FAH = false auto / should-escalate)

| Policy | Auto-handle | FAH | Escalate F1 |
| --- | ---: | ---: | ---: |
| Always escalate | 0.000 | 0.000 | 0.730 |
| Confidence thr 0.7 | 0.375 | 0.383 | 0.592 |
| Proposed risk-aware | 0.260 | 0.165 | 0.730 |

### Replies (heuristic judge means)

| System | Correctness | Groundedness | Helpfulness | Hallucination safety |
| --- | ---: | ---: | ---: | ---: |
| Generic | 3.70 | 5.00 | 4.00 | 5.00 |
| Nearest case | 3.68 | 4.96 | 3.59 | 4.96 |
| ResolveFlow offline | 3.70 | 5.00 | 4.00 | 5.00 |

### Main table

| System | Intent Macro-F1 | Escalation F1 | Auto-Handle | FAH | Reply Correctness | Groundedness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.012 | 0.730 | 0.000 | 0.000 | 3.69 | 5.00 |
| TF-IDF + LR | 0.683 | 0.592 | 0.375 | 0.383 | — | — |
| Nearest Case | — | — | — | — | 3.68 | 4.96 |
| ResolveFlow | 0.683 | 0.730 | 0.260 | 0.165 | 3.69 | 5.00 |

Artifacts: `artifacts/final/`.

---

## 8. Failure Analysis

Top modes by frequency × severity (see `report/failure_analysis.md`):

1. **False auto-handle** (19) — policy allows automation on gold-escalate cases.
2. **Intent errors (other)** (26) — lexical confusion beyond named pairs.
3. **Ambiguous / unclear** (19) — thin tweets; abstention vs forced class.
4. **refund_status → refund_request** (5) — taxonomy boundary.
5. **delivery_delay ↔ missing package** (3+) — topic-level retrieval/classifier confusion.

---

## 9. What Is Misleading About My Headline Number?

**Safe Auto-Handling Rate = 16.5%** is operationally meaningful but easy to over-read.

1. **Golden set size (200)** — small; CI on Macro-F1 spans ~0.62–0.75.
2. **Stratified sampling** — not natural production traffic mix.
3. **One brand (AmazonHelp)** — policies and language differ elsewhere.
4. **Twitter CS corpus** — not modern in-app chat or authenticated sessions.
5. **Offline evaluation** — reported agent uses TF-IDF + templates, not hosted GPT; OpenAI path unmeasured here.
6. **Judge bias** — heuristic `judge_v1` saturates groundedness/hallucination on templates; length–score corr ≈ 0.12.
7. **Human calibration (n=40, solo)** — agreement looks high partly because variance is low and ratings are rubric-aligned; Spearman is uninformative when scores are nearly constant; no second rater.
8. **Threshold/policy choices** — tuned with development/silver signals; golden is final report only, but design still influenced by non-golden data.
9. **No live feedback** — no CSAT, AHT, repeat contact, or true containment.
10. **No account access** — system cannot verify orders/payments; escalation is often the correct ceiling.

Treat the headline as a **conservative containment estimate under this rubric**, not production containment.

---

## 10. One More Week

1. **Hybrid retrieval + rerank** — fix topic-similar / resolution-different neighbors (refund request vs status; late vs missing).
2. **Calibrate escalation on labeled validation** — constrain FAH ≤ target while maximizing safe auto-handle.
3. **Expand golden + second annotator** — especially hard/refund boundary cases; independent reply ratings via Streamlit.
4. **Run OpenAI classifier/responder** with frozen prompts and compare ablations under the same judge.
5. **Structured policy layer** — explicit allow/deny actions instead of history-only grounding.

---

## 11. Conclusion

ResolveFlow shows that for AmazonHelp tweets, a classical intent head already beats majority by a large margin, retrieval provides grounding evidence, and a risk-aware gate cuts false auto-handling versus confidence-only automation—while still only safely auto-handling ~17% of golden cases under a strict definition. Trust comes from leakage controls, baselines, safety tests, and honest limits—not from a single judge score.
