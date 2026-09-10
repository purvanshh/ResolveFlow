# ResolveFlow — AI Support Agent for AmazonHelp

## 1. Executive Summary

ResolveFlow is a grounded customer-support agent prototype for **AmazonHelp** tweets. It classifies intent (12-class brand taxonomy), retrieves historical cases, drafts a reply with **gpt-4o-mini**, runs safety checks, and applies a risk-aware escalation policy.

**Headline metric: Safe Auto-Handling Rate = 0.255**  
**Definition:** fraction of golden examples that were **auto-handled**, with **correct intent**, **no safety / unsupported-claim flags**, and **gold also labeled auto_handle**  
(= 51 / 200 on the frozen set).

Do **not** read **Auto-handle rate = 0.440** as “44% of queries can safely be automated.” That is only the share of cases the policy chose not to escalate. Safe Auto-Handling Rate is the stricter intersection with intent correctness, safety, and gold agreement.

Supporting numbers (`gpt-4o-mini` classifier + responder + MiniLM retrieval + `gpt-4o-mini` judge):

| Metric | Value |
| --- | ---: |
| Intent Macro-F1 (LLM) | 0.669 (95% bootstrap CI [0.598, 0.727]) |
| Intent Macro-F1 (TF-IDF baseline) | **0.683** |
| Auto-handle rate | 0.440 |
| False auto-handle rate (among should-escalate) | 0.243 |
| Escalation F1 | 0.767 |
| Retrieval Recall@3 | 0.565 |
| Reply correctness / groundedness | 4.36 / 4.33 |
| Unsupported-claim rate | 0.015 |
| Safety suite | 5/6 PASS (GPT fail: `account_specific`; DeepSeek `fake_policy` was harness FP → 6/6 after regrade) |

**Honest takeaway:** on this golden set the classical TF-IDF head still slightly leads LLM intent Macro-F1 (CIs overlap). ResolveFlow’s value is stronger replies vs generic/nearest baselines, higher safe auto-handling than the offline template agent, and a calibrated escalation story—not a claim that GPT uniquely wins classification.

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
| Proposed risk-aware | 0.440 | 0.243 | **0.767** |

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
| ResolveFlow | 0.669 | 0.767 | 0.440 | 0.243 | 4.36 | 4.33 |

Artifacts: `artifacts/final/`.

---

## 7b. LLM Model Comparison

Same frozen golden set, retrieval, safety, escalation, and **gpt-4o-mini** judge. DeepSeek uses API model `deepseek-flash` (version DeepSeek-V4.1-Flash). Thinking uses `extra_body={"thinking":{"type":"enabled"}}` + `reasoning_effort=high`; non-thinking uses `type=disabled`.

| Metric | TF-IDF | GPT-4o-mini | DeepSeek Non-thinking | DeepSeek Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | **0.683** | 0.669 | 0.653 | 0.634 |
| Safe Auto-Handling Rate | — | **0.255** | **0.255** | **0.255** |
| False Auto-handle | — | **0.243** | 0.270 | 0.322 |
| Escalation F1 | — | **0.767** | 0.760 | 0.726 |
| Reply correctness | — | 4.36 | **4.74** | **4.78** |
| Reply groundedness | — | 4.33 | **4.74** | **4.77** |
| Unsupported-claim rate | — | **0.015** | 0.020 | 0.020 |
| Safety suite | — | 5/6 (`account_specific`) | **6/6**† | **6/6**† |
| Mean latency (s) | — | — | **1.97** | 6.98 |

† DeepSeek originally recorded 5/6 on `fake_policy` due to a brittle substring harness (`forbid_in_reply: always guaranteed`) that false-positive’d safe denials. After negation-aware grading on the **same stored replies**, DeepSeek is 6/6. That is a harness correction, not a model-behavior change. GPT-4o-mini still fails `account_specific` (separate issue).

**Interpretation (n=200):** safe auto-handle is tied. GPT-4o-mini is better on false auto-handle and escalation F1. DeepSeek scores higher on judged reply quality. Thinking mode does not improve safe automation vs non-thinking and increases latency ~3.5× with worse FAH. Intent Macro-F1 differences GPT vs DeepSeek-NT have overlapping bootstrap CIs; TF-IDF remains strongest on intent alone.

Separately, the deterministic policy validator was strengthened to catch positive unsupported policy assertions (e.g. `"Amazon's policy guarantees you a refund."`) without treating historical tweets as authoritative policy docs, while preserving safe negations. Re-checking GPT golden drafts found **no** decision flips from that change; Safe Auto-Handling Rate remains **0.255**.

**Model selection:** keep **GPT-4o-mini** as the default ResolveFlow LLM. Treat DeepSeek non-thinking as a strong alternative if optimizing judge reply scores; do not default to thinking mode.

See `artifacts/final/model_comparison.md` and `artifacts/final/model_failure_comparison.json`.

---

## 8. Failure Analysis

Top modes (see `report/failure_analysis.md` and refreshed rankings):

1. **False auto-handle** — still ~24% of should-escalate cases.
2. **Intent confusion** — refund status/request; delivery vs missing package.
3. **Ambiguous / thin tweets** — abstention vs forced class.
4. **Unsupported claims** — ~1.5% of drafts (no longer zero with free-form LLM replies).
5. **Safety suite miss (GPT `account_specific`)** — GPT escalated correctly and did not claim account access, but the suite still failed on a brittle `forbid_in_reply` substring (`your last transaction` inside user-directed “check … for your last transaction”). Separate from DeepSeek `fake_policy` (harness FP, now 6/6 after regrade). See failure analysis §7.

---

## 9. What Is Misleading About My Headline Number?

The 25.5% Safe Auto-Handling Rate is measured on a frozen 200-example golden set sampled from one brand's historical Twitter support data. It is **not** an estimate of production automation coverage. The sample may not represent current traffic, and the metric depends on the chosen escalation policy, taxonomy, safety validator, and labeling decisions. In particular, a conservative system can improve safety by escalating more cases, so the number should be interpreted together with escalation quality and false-auto-handle rate.

Also keep these distinctions in view:

1. **Auto-handle rate (0.440) ≠ Safe Auto-Handling Rate (0.255).** The former is “policy chose not to escalate”; the latter requires correct intent, clean safety flags, and gold agreement on auto-handle (51/200).
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
12. **Safety suite not perfect** — GPT OpenAI replies still 5/6 (`account_specific`).

Treat the headline as a **conservative containment estimate under this rubric**, not production containment.

---

## 10. One More Week

1. **Hybrid retrieval + rerank** — fix topic-similar / resolution-different neighbors.
2. **Calibrate escalation on labeled validation** — constrain FAH ≤ target while maximizing safe auto-handle.
3. **Expand golden + second annotator** — independent Streamlit ratings; tighten refund boundaries.
4. **Strengthen account-specific safety gate** — fix the failing suite case without killing helpful drafts.
5. **Structured policy layer** — explicit allow/deny actions instead of history-only grounding.

---

## 11. Conclusion

ResolveFlow with **gpt-4o-mini** shows useful automation for AmazonHelp tweets: stronger judged replies than generic/nearest baselines, better escalation F1 than confidence-only policies, and a measurable safe auto-handle rate of **25.5%**. It does **not** overturn TF-IDF on intent Macro-F1 on this frozen set. Trust comes from leakage controls, baselines, ablations, and honest limits—not from a single judge score.
