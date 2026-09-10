# Decision Log

Real decisions for ResolveFlow (AmazonHelp). Ordered roughly by impact.

### Decision 1 — Single brand: AmazonHelp
**Why:** Brand-specific resolution style and intent mix; ~82.5k usable conversations with median length 3. Multi-brand would dilute taxonomy and retrieval.

### Decision 2 — Conversation-level split + golden exclusion from retrieval
**Why:** Prevents thread/text leakage into baselines and the historical case index. Final audit: 0 conversation / exact / normalized / retrieval overlaps.

### Decision 3 — 12 intents including `other_unclear`
**Why:** Covers observed AmazonHelp clusters without Banking77’s generic banking set. `other_unclear` supports abstention instead of forced labels.

### Decision 4 — Frozen golden set (n=200) with checksum
**Why:** Stops evaluation drift. Manifest SHA-256 `e974255a5ab6e06516e11c7c9e74b61b53c81d2c10a5386a7d549e951bb4ef36`. Status FROZEN.

### Decision 5 — Solo + rule-assisted labeling (not bulk LLM labels)
**Why:** LLM labels would circularly validate LLM systems. Rules assist consistency; humans own hard boundaries (especially refund request vs status).

### Decision 6 — TF-IDF + logistic regression as primary intent baseline/offline head
**Why:** Strong, cheap, interpretable classical bar (Macro-F1 0.683). Offline final numbers reuse this head so results are reproducible without API keys.

### Decision 7 — MiniLM embeddings for historical retrieval (k=3, threshold 0.45)
**Why:** Semantic neighbors for grounding. k=3 chosen after ablation: k=0 blocks auto-handle; k=1/3/5 similar automation; intent F1 unchanged with k.

### Decision 8 — Grounded template responder offline; LLM responder optional
**Why:** Without API key, templates + evidence still demonstrate the safety/escalation story. Avoids fabricating hosted-LLM metrics.

### Decision 9 — Conservative risk-aware escalation over confidence thresholds
**Why:** Confidence-only FAH among should-escalate ≈ 0.383 vs proposed ≈ 0.165. High-risk intents (payment, refunds, account, missing package) always escalate.

### Decision 10 — Prioritize false auto-handle among should-escalate
**Why:** Operational safety metric. “Among auto-handled” FAH can look different; Phase 5 reporting uses the should-escalate denominator.

### Decision 11 — Heuristic judge with cache + Streamlit human UI
**Why:** LLM-as-judge without calibration is weak. Cache avoids spend. Streamlit `rate_replies.py` supports independent human ratings; current n=40 is solo calibration.

### Decision 12 — Headline = Safe Auto-Handling Rate (not Macro-F1 alone)
**Why:** Intent Macro-F1 equals TF-IDF offline—doesn’t show agent value. Safe auto-handle rate encodes intent + safety + agreement with gold automation.

### Decision 13 — No Banking77 / no multi-brand transfer learning
**Why:** Brand-specific taxonomy is more relevant to Hiver-style support automation than generic banking intents.

### Decision 14 — Safety suite as gate, not afterthought
**Why:** Injection, unsupported refunds, fake policies, and account-specific asks must escalate without inventing actions. Suite 6/6 PASS offline.

### Decision 15 — Report “What is misleading about my headline”
**Why:** Small golden set, offline mode, judge saturation, and no live CSAT would otherwise invite overconfidence.

### Decision 16 — Benchmark DeepSeek-V4.1-Flash thinking vs non-thinking
**Hypothesis:** Substituting `deepseek-flash` (DeepSeek-V4.1-Flash) into the same pipeline may match/beat GPT-4o-mini on safe automation; thinking mode may help hard cases.
**Controls:** frozen golden n=200; same taxonomy/prompts/retrieval k=3/safety/escalation; judge=`gpt-4o-mini`; thinking via API `extra_body.thinking.type` (not prompt simulation); isolated artifact dirs.
**Results (measured):**
- Safe auto-handle: GPT=DeepSeek-NT=DeepSeek-T=**0.255**
- FAH among should-escalate: GPT **0.243** < NT 0.270 < T 0.322
- Escalation F1: GPT **0.767** ≈ NT 0.760 > T 0.726
- Reply correctness/groundedness: DeepSeek higher (~4.74–4.78) than GPT (~4.33–4.36)
- Intent Macro-F1: TF-IDF 0.683 > GPT 0.669 > NT 0.653 > T 0.634 (CIs overlap GPT vs NT)
- Latency: NT ~2.0s vs T ~7.0s mean
- Safety suite: all LLMs 5/6
**Decision:** Keep **GPT-4o-mini** as default (better FAH + escalation F1 at tied safe auto-handle). Prefer DeepSeek **non-thinking** over thinking. Do not enable thinking mode by default.
