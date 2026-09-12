# ResolveFlow — AI Support Agent for AmazonHelp

## 1. Problem Framing

ResolveFlow is a **conservative** AI support system for **AmazonHelp** tweets: LLM intent classification and grounded reply drafting, plus deterministic safety validation and escalation controls outside the model.

Good is **useful automation that refuses unsupported claims and escalates when account-specific or high-risk**. Maximizing auto-handle rate alone is the wrong objective. False auto-handling is treated as the primary safety failure.

**Not built:** live account access, refunds/CRM actions, production deployment, multi-brand support, real-time Twitter integration.

---

## 2. Data & Taxonomy

- Dataset: Customer Support on Twitter (`twcs.csv`); brand **AmazonHelp** (see [`CITATIONS.md`](../CITATIONS.md)).
- Taxonomy: 12 intents (`configs/intents.yaml`), including `other_unclear` for abstention.
- Golden set: **200** frozen examples; checksum `e974255a…bb4ef36` (**labels and examples unchanged**).
- Splits: conversation-level; golden excluded from development messages and retrieval corpus.
- Final leakage audit: conversation / exact / normalized / retrieval overlap = **0** (`STATUS: PASS`).
- Final eval mode: **openai** (`RESOLVEFLOW_LLM_MODEL=gpt-4o-mini`).

---

## 3. Architecture

```text
Customer message
   ↓
Intent (gpt-4o-mini / classifier_v1)
   ↓
Retrieve historical cases (MiniLM, k=3, rerank_mode=none)
   ↓
Draft response (gpt-4o-mini)
   ↓
Deterministic safety validation
   ↓
Risk-aware escalation (high-risk intents + detect_message_risk)
   ↓
Auto-handle  /  Human escalate
```

---

## 4. Evaluation Methodology

- Frozen golden set + automated intent/escalation metrics.
- Baselines: majority, TF-IDF+LR, always-escalate, confidence threshold, nearest historical reply, generic reply.
- Reply judge (`judge_v1`) via **gpt-4o-mini**, cached under `artifacts/final/judge_cache.json` (assignment requires an LLM-as-judge rubric).
- Human calibration: 40 examples, solo annotator; Streamlit UI `scripts/rate_replies.py`.
- Ablations: retrieval k∈{0,1,3,5}; escalation policy comparison; optional retrieval rerank (not adopted).
- Safety suite: unsupported refunds, injection, fake policy, account-specific, ambiguous, false action.
- Headline CIs: percentile bootstrap (n_boot=500, seed=42) on frozen GPT predictions — `scripts/compute_headline_cis.py` → `artifacts/final/headline_confidence_intervals.json`.

**Policy-selection caveat:** escalation policy and message-level risk cues were selected using analysis of this same frozen 200-example evaluation set. Golden **labels and examples were never changed**, and the SAH/FAH metric definitions were not redefined to chase a number—but policy selection on the eval set still risks **overfitting** and can make reported FAH/SAH look optimistic relative to a held-out calibration split.

---

## 5. Baselines & Final Results

**Headline metric: Safe Auto-Handling Rate = 0.255** (= 51/200)  
**Definition:** auto-handled **and** correct intent **and** no safety flags **and** gold also `auto_handle`.

Do **not** read **Auto-handle = 0.320** as safe automation, and do **not** read **SAH = 25.5%** as production coverage.

| Metric | Point estimate | 95% bootstrap CI |
| --- | ---: | ---: |
| Safe Auto-Handling | **25.5%** (51/200) | [19.5%, 31.0%] |
| Auto-handle | **32.0%** (64/200) | [26.2%, 38.5%] |
| FAH (among should-escalate) | **4.3%** (5/115) | [0.9%, 8.9%] |
| Escalation F1 | **0.876** | [0.831, 0.916] |
| Intent Macro-F1 (LLM) | **0.669** | [0.598, 0.727] |
| Intent Macro-F1 (TF-IDF) | **0.683** | — |
| Safety suite | **6/6** | — |

Intervals quantify resampling variability on the frozen set; they do **not** remove policy-selection bias (see §4).

**Central engineering lesson:** TF-IDF slightly beats GPT on intent Macro-F1 (0.683 vs 0.669), but TF-IDF-only and hybrid routing did **not** improve the end-to-end safety/automation operating point. The largest operational gain came from conservative escalation controls and message-level risk detection: **FAH fell from 24.3% → 4.3% while SAH remained 25.5%** (with the policy-selection caveat above).

### Intent

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Majority | 0.075 | 0.012 | 0.010 |
| TF-IDF + LR | 0.695 | **0.683** | 0.676 |
| ResolveFlow (gpt-4o-mini) | 0.655 | 0.669 | 0.660 |

### Escalation (FAH = false auto / should-escalate)

| Policy | Auto-handle | FAH | Escalate F1 |
| --- | ---: | ---: | ---: |
| Always escalate | 0.000 | 0.000 | 0.730 |
| Confidence thr 0.7 | 0.375 | 0.383 | 0.592 |
| Risk-aware (pre-calibration) | 0.440 | 0.243 | 0.767 |
| Risk-aware + `order_quality_issue` high-risk | 0.350 | 0.096 | 0.849 |
| + message-level risk cues (adopted) | **0.320** | **0.043** | **0.876** |

Adopted path (frozen predictions, policy recomputed): FAH **24.3% → 9.6% → 4.3%**; SAH **25.5% → 25.5% → 25.5%**; Escalation F1 **0.767 → 0.849 → 0.876**.

### Replies (gpt-4o-mini judge means — directional)

| System | Correctness | Groundedness | Helpfulness | Hallucination safety |
| --- | ---: | ---: | ---: | ---: |
| Generic | 3.24 | 2.38 | 2.29 | 4.63 |
| Nearest case | 3.46 | 3.42 | 3.16 | 4.29 |
| ResolveFlow | 4.36 | 4.33 | 4.04 | 4.94 |

On the LLM judge’s scoring, ResolveFlow received higher soft reply-quality scores than generic/nearest baselines; **human agreement for these soft quality dimensions was weak**, so treat these as **directional**, not validated quality estimates (see Judge validity).

### Main operating-point table

| System | Intent Macro-F1 | Escalation F1 | Auto-Handle | FAH |
| --- | ---: | ---: | ---: | ---: |
| Majority | 0.012 | 0.730 | 0.000 | 0.000 |
| TF-IDF + LR | 0.683 | 0.592 | 0.375 | 0.383 |
| ResolveFlow | 0.669 | 0.876 | 0.320 | 0.043 |

---

## 6. Judge Validity

The LLM judge was calibrated against **40** human-rated examples (`artifacts/final/judge_human_agreement.json`). Agreement:

| Dimension | Spearman | Weighted κ |
| --- | ---: | ---: |
| correctness | 0.299 | 0.147 |
| groundedness | 0.215 | 0.153 |
| helpfulness | **−0.191** | **−0.051** |
| overall | **−0.003** | **−0.130** |
| hallucination_safety | **0.679** | **0.792** |

Correctness / groundedness / helpfulness / overall are **weakly human-validated** (helpfulness and overall are near-zero or negative). Therefore judge scores are **directional evidence** for response-quality comparisons, **not** ground truth. The strongest validated conclusion from the judge is the **hallucination/safety** dimension.

The judge remains useful because: (1) it gives a consistent automated comparison across systems; (2) hallucination/safety showed meaningful human agreement; (3) soft quality dimensions stay exploratory; (4) results must not be read as objective quality rankings.

---

## 7. LLM Model Comparison

Same frozen golden set, retrieval, safety, escalation, and **gpt-4o-mini** judge.

**DeepSeek identifier evidence:** API model string **`deepseek-flash`**, base URL `https://api.deepseek.com`, repo `model_version` label **`DeepSeek-V4.1-Flash`** (matches current DeepSeek pricing docs for that API id). Thinking: `extra_body.thinking.type=enabled` + `reasoning_effort=high`. Artifacts dated **2026-09-10** locally. No per-response immutable snapshot id is stored; see [`CITATIONS.md`](../CITATIONS.md).

| Metric | TF-IDF | GPT-4o-mini | DeepSeek Non-thinking | DeepSeek Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | **0.683** | 0.669 | 0.653 | 0.634 |
| Auto-handle rate | — | 0.320 | 0.355 | 0.365 |
| Safe Auto-Handling Rate | — | **0.255** | **0.255** | **0.255** |
| False Auto-handle | — | **0.043** | 0.087 | 0.096 |
| Escalation F1 | — | **0.876** | 0.861 | 0.860 |
| Reply correctness (judge) | — | 4.36 | 4.74 | 4.78 |
| Reply groundedness (judge) | — | 4.33 | 4.74 | 4.77 |
| Unsupported-claim rate | — | **0.015** | 0.020 | 0.020 |
| Safety suite | — | **6/6**† | **6/6**† | **6/6**† |
| Mean latency (s) | — | — | **1.97** | 6.98 |

† Safety harness regrades on stored replies. Escalation includes `order_quality_issue` high-risk + message risk cues; intents/replies frozen. SAH remains **0.255**.

**Interpretation:** SAH tied. GPT has the lowest FAH and highest escalation F1 after risk-cue adoption (objective/automated metrics). On the LLM judge’s scoring, DeepSeek received higher soft reply-quality scores; human agreement for those dimensions was weak, so treat that comparison as **directional**, not a validated quality win. TF-IDF still leads intent Macro-F1. Keep **GPT-4o-mini** as default.

---

## 8. Safety / Evaluation Integrity

Two suite failures (`fake_policy`, `account_specific`) were **harness false positives** from substring matching; stored model replies were unchanged. Assertion-aware grading + a strengthened deterministic validator → suite **6/6**. This is evaluation-integrity work, **not** “the model became safer.”

---

## 9. Failure Analysis

1. **False auto-handle** — **5/115** remaining (boundary/intent ambiguity; not retrieval): ambiguous FR delay; carrier-attempt missing; refuse-return; thanks/return-label; payment-as-cancel.
2. **Intent confusion** — refund status/request; delivery vs missing package (Macro-F1 still trails TF-IDF).
3. **Ambiguous / thin tweets** — abstention vs forced class.
4. **Unsupported claims** — ~1.5% of drafts.
5. **Safety suite harness FPs (fixed)** — suite **6/6**.

Details: `report/failure_analysis.md`.

---

## 10. What Is Misleading About My Headline Number?

The 25.5% Safe Auto-Handling Rate is measured on a frozen 200-example golden set. It does **not** mean “25.5% of production customer requests can safely be automated.” It depends on sampling, taxonomy, labels, escalation policy, safety validator, and evaluation definitions.

Also:

1. **Auto-handle (0.320) ≠ SAH (0.255).**
2. **Policy selection on the golden set** can make FAH/SAH optimistic even though labels/examples stayed frozen.
3. **Judge soft-quality scores are weakly validated** (see §6); do not equate them with production reply quality.
4. **One brand, Twitter CS corpus, n=200** — not live authenticated support.
5. **LLM does not beat TF-IDF on intent Macro-F1** on this set.
6. **No live CSAT / account APIs.**

Treat the headline as a **conservative containment estimate under this rubric**, not production containment.

---

## 11. One More Week

Future experiments only (not implemented):

1. Intent boundaries for the remaining five FAH cases.
2. More boundary-focused golden examples if annotation budget allows.
3. Calibrate confidence on a **held-out** validation set (reduce policy-selection bias).
4. Principled ensemble only if calibrated confidence exists.
5. Resolution-aware retrieval on a larger eval set.
6. Multi-brand generalization.

---

## 12. Decision Log

Engineering decisions (taxonomy size, `other_unclear`, deterministic safety outside the LLM, conservative escalation, rejecting rerank/TF-IDF hybrid, freezing residual FAH, etc.): [`report/decision_log.md`](decision_log.md).

**Conclusion:** ResolveFlow with **gpt-4o-mini** shows a measurable safe auto-handle rate of **25.5%** and strong escalation F1 after conservative controls. It does **not** overturn TF-IDF on intent Macro-F1. Soft reply-quality judge rankings are directional only. Trust comes from leakage controls, baselines, ablations, CIs, and honest limits—not from a single judge score.
