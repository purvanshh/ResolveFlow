# ResolveFlow — AI Support Agent for AmazonHelp

## 1. Executive Summary

ResolveFlow is a grounded customer-support agent prototype for **AmazonHelp** tweets. It classifies intent (12-class brand taxonomy), retrieves historical cases, drafts a reply with **gpt-4o-mini**, runs safety checks, and applies a risk-aware escalation policy.

**Headline metric: Safe Auto-Handling Rate = 0.255**  
**Definition:** fraction of golden examples that were **auto-handled**, with **correct intent**, **no safety / unsupported-claim flags**, and **gold also labeled auto_handle**  
(= 51 / 200 on the frozen set).

Do **not** read **Auto-handle rate = 0.320** as “32% of queries can safely be automated.” That is only the share of cases the policy chose not to escalate. Safe Auto-Handling Rate is the stricter intersection with intent correctness, safety, and gold agreement.

Supporting numbers (`gpt-4o-mini` + MiniLM retrieval + calibrated escalation with `order_quality_issue` high-risk **and** message-level risk cues):

| Metric | Value |
| --- | ---: |
| Intent Macro-F1 (LLM) | 0.669 (95% bootstrap CI [0.598, 0.727]) |
| Intent Macro-F1 (TF-IDF baseline) | **0.683** |
| Auto-handle rate | 0.320 |
| False auto-handle rate (among should-escalate) | **0.043** |
| Escalation F1 | **0.876** |
| Retrieval Recall@3 | 0.565 |
| Reply correctness / groundedness | 4.36 / 4.33 |
| Unsupported-claim rate | 0.015 |
| Safety suite | **6/6 PASS** |

**Honest takeaway:** TF-IDF still slightly leads LLM intent Macro-F1. ResolveFlow’s value is grounded replies plus calibrated escalation (high-risk intents + message risk cues) that cut FAH without lowering SAH—not a claim that GPT uniquely wins classification.

---

## 2. Problem Framing

Good is **useful automation that refuses unsupported claims and escalates when account-specific or high-risk**. Maximizing auto-handle rate alone is the wrong objective. False auto-handling is treated as the primary safety failure.

---

## 3. Scope

**Built:** intent classification, historical retrieval, grounded reply drafting, escalation policy, safety gates, baselines, frozen golden evaluation, LLM judge + human calibration scaffold, failure analysis.

**Not built:** live account access, refunds/CRM actions, production deployment, multi-brand support, real-time Twitter integration.

---

## 4. Data & Methodology

- Dataset: Customer Support on Twitter (`twcs.csv`); brand **AmazonHelp**.
- Taxonomy: 12 intents (`configs/intents.yaml`), including `other_unclear` for abstention.
- Golden set: **200** frozen examples; checksum `e974255a…bb4ef36`.
- Splits: conversation-level; golden excluded from development messages and retrieval corpus.
- Final leakage audit: conversation / exact / normalized / retrieval overlap = **0** (`STATUS: PASS`).
- Final eval mode: **openai** (`RESOLVEFLOW_LLM_MODEL=gpt-4o-mini`).

---

## 5. Architecture

```text
Customer message
   ↓
Intent (gpt-4o-mini)
   ↓
Retrieve historical cases (MiniLM, k=3)
   ↓
Draft response (gpt-4o-mini)
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
- Reply judge (`judge_v1`) via **gpt-4o-mini**, cached under `artifacts/final/judge_cache.json`.
- Human calibration: 40 examples, solo annotator; Streamlit UI `scripts/rate_replies.py`.
- Ablations: retrieval k∈{0,1,3,5}; escalation policy comparison.
- Safety suite: unsupported refunds, injection, fake policy, account-specific, ambiguous, false action.

---

## 7. Results

### Intent

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Majority | 0.075 | 0.012 | 0.010 |
| TF-IDF + LR | 0.695 | **0.683** | 0.676 |
| ResolveFlow (gpt-4o-mini) | 0.655 | 0.669 | 0.660 |

### Difficulty (accuracy / macro-F1 over intents present)

| Difficulty | n | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Easy | 58 | 0.672 | 0.491 |
| Medium | 128 | 0.648 | 0.603 |
| Hard | 14 | 0.643 | 0.557 |

### Escalation (FAH = false auto / should-escalate)

| Policy | Auto-handle | FAH | Escalate F1 |
| --- | ---: | ---: | ---: |
| Always escalate | 0.000 | 0.000 | 0.730 |
| Confidence thr 0.7 | 0.375 | 0.383 | 0.592 |
| Risk-aware (pre-calibration) | 0.440 | 0.243 | 0.767 |
| Risk-aware + `order_quality_issue` high-risk | 0.350 | 0.096 | 0.849 |
| + message-level risk cues (adopted) | **0.320** | **0.043** | **0.876** |

### Replies (gpt-4o-mini judge means)

| System | Correctness | Groundedness | Helpfulness | Hallucination safety |
| --- | ---: | ---: | ---: | ---: |
| Generic | 3.24 | 2.38 | 2.29 | 4.63 |
| Nearest case | 3.46 | 3.42 | 3.16 | 4.29 |
| ResolveFlow | **4.36** | **4.33** | **4.04** | **4.94** |

### Retrieval k ablation

| k | Auto-handle | FAH (among auto) | Unsupported claim |
| --- | ---: | ---: | ---: |
| 0 | 0.000 | 0.000 | 0.022 |
| 1 | 0.445 | 0.326 | 0.005 |
| 3 | 0.440 | 0.318 | 0.015 |
| 5 | 0.450 | 0.333 | 0.010 |

### Main table

| System | Intent Macro-F1 | Escalation F1 | Auto-Handle | FAH | Reply Correctness | Groundedness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority | 0.012 | 0.730 | 0.000 | 0.000 | 3.24 | 2.38 |
| TF-IDF + LR | 0.683 | 0.592 | 0.375 | 0.383 | — | — |
| Nearest Case | — | — | — | — | 3.46 | 3.42 |
| ResolveFlow | 0.669 | 0.876 | 0.320 | 0.043 | 4.36 | 4.33 |

Artifacts: `artifacts/final/`.

---

## 7b. LLM Model Comparison

Same frozen golden set, retrieval, safety, escalation, and **gpt-4o-mini** judge. DeepSeek uses API model `deepseek-flash` (version DeepSeek-V4.1-Flash). Thinking uses `extra_body={"thinking":{"type":"enabled"}}` + `reasoning_effort=high`; non-thinking uses `type=disabled`.

| Metric | TF-IDF | GPT-4o-mini | DeepSeek Non-thinking | DeepSeek Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | **0.683** | 0.669 | 0.653 | 0.634 |
| Auto-handle rate | — | 0.320 | 0.355 | 0.365 |
| Safe Auto-Handling Rate | — | **0.255** | **0.255** | **0.255** |
| False Auto-handle | — | **0.043** | 0.087 | 0.096 |
| Escalation F1 | — | **0.876** | 0.861 | 0.860 |
| Reply correctness | — | 4.36 | **4.74** | **4.78** |
| Reply groundedness | — | 4.33 | **4.74** | **4.77** |
| Unsupported-claim rate | — | **0.015** | 0.020 | 0.020 |
| Safety suite | — | **6/6**† | **6/6**† | **6/6**† |
| Mean latency (s) | — | — | **1.97** | 6.98 |

† Safety harness regrades on stored replies. Escalation recomputed with `order_quality_issue` high-risk **plus** message-level risk cues; frozen intents/replies. SAH remains **0.255**.

**Interpretation (n=200):** SAH tied. GPT has the lowest FAH and highest escalation F1 after risk-cue adoption. DeepSeek wins judged reply quality. TF-IDF still leads intent Macro-F1.

**Escalation stack:** (1) high-risk intents including `order_quality_issue`; (2) message risk cues for refund/money-back, damaged/defective/wrong item, never-received, fraud. Hybrid GPT+TF-IDF not adopted (SAH regression despite higher Macro-F1). See `artifacts/final/intent_risk_experiment.json`.

**Model selection:** keep **GPT-4o-mini** as the default ResolveFlow LLM.

See `artifacts/final/model_comparison.md` and `artifacts/final/model_failure_comparison.json`.

---

## 8. Failure Analysis

Top modes (see `report/failure_analysis.md` and refreshed rankings):

1. **False auto-handle** — now **~4.3%** of should-escalate after high-risk intents + message risk cues (was 24.3% → 9.6% → 4.3%). Remaining 5 FAH lack narrow lexical cues.
2. **Intent confusion** — refund status/request; delivery vs missing package (Macro-F1 still trails TF-IDF).
3. **Ambiguous / thin tweets** — abstention vs forced class.
4. **Unsupported claims** — ~1.5% of drafts.
5. **Safety suite harness FPs (fixed)** — suite **6/6**.

---

## 9. What Is Misleading About My Headline Number?

The 25.5% Safe Auto-Handling Rate is measured on a frozen 200-example golden set sampled from one brand's historical Twitter support data. It is **not** an estimate of production automation coverage. The sample may not represent current traffic, and the metric depends on the chosen escalation policy, taxonomy, safety validator, and labeling decisions. In particular, a conservative system can improve safety by escalating more cases, so the number should be interpreted together with escalation quality and false-auto-handle rate.

Also keep these distinctions in view:

1. **Auto-handle rate (0.320) ≠ Safe Auto-Handling Rate (0.255).** The former is “policy chose not to escalate”; the latter requires correct intent, clean safety flags, and gold agreement on auto-handle (51/200).
2. **Golden set size (200)** — CI on Macro-F1 spans ~0.60–0.73.
3. **Stratified sampling** — not natural production traffic mix.
4. **One brand (AmazonHelp)** — policies and language differ elsewhere.
5. **Twitter CS corpus** — not modern in-app chat or authenticated sessions.
6. **LLM ≠ TF-IDF win** — headline is *not* “GPT beats classical ML on intent.”
7. **Judge bias / weak calibration** — correctness Spearman ≈ 0.30 vs solo human sample; within-1 overall is high but exact agreement is modest.
8. **Human calibration (n=40, solo)** — no second rater; Streamlit UI exists for independent ratings.
9. **Threshold/policy choices** — influenced by development/silver signals.
10. **No live feedback** — no CSAT, AHT, repeat contact, or true containment.
11. **No account access** — system cannot verify orders/payments; escalation is often the correct ceiling.
12. **Safety suite** — currently 6/6 after harness fixes; still not a substitute for live account APIs or production red-teaming.

Treat the headline as a **conservative containment estimate under this rubric**, not production containment.

---

## 10. One More Week

1. **Intent quality on residual FAH** — resolution-aware rerank was tested and did **not** move SAH/FAH; remaining 11 FAH are wrong-intent autos into non–high-risk classes.
2. **Fraud / payment language gates** — escalate when message cues conflict with a soft predicted intent.
3. **Expand golden + second annotator** — independent Streamlit ratings; tighten refund boundaries.
4. **Structured policy layer** — explicit allow/deny actions instead of history-only grounding.
5. **Reply-grounding quality** — optional rerank may still help drafts even when escalation is unchanged (not measured with a new judge pass here).

---

## 11. Conclusion

ResolveFlow with **gpt-4o-mini** shows useful automation for AmazonHelp tweets: stronger judged replies than generic/nearest baselines, better escalation F1 than confidence-only policies, and a measurable safe auto-handle rate of **25.5%**. It does **not** overturn TF-IDF on intent Macro-F1 on this frozen set. Trust comes from leakage controls, baselines, ablations, and honest limits—not from a single judge score.
