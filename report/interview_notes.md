# Interview Notes

### Why this brand?
AmazonHelp had large usable volume (~82.5k conversations), short threads (median 3), and clear retail support patterns (delivery, refunds, Prime, account).

### Why these intents?
Derived from AmazonHelp clustering + manual cleanup into 12 operational labels. Included `other_unclear` so the system can abstain.

### Why conversation-level splitting?
Prevents the same thread appearing in train/retrieval and golden evaluation. Leakage script checks IDs and normalized text; final audit PASS with 0 overlaps.

### Why TF-IDF?
Strong classical baseline (Macro-F1 0.683). Interpretable and reproducible offline. Final reported classifier *is* TF-IDF because no API key was used for the fingerprint.

### Why embeddings / retrieval?
Ground replies in historical AmazonHelp resolutions; provide evidence for the judge and safety story. Recall@1/3/5 = 0.310 / 0.565 / 0.670.

### Why not just an LLM?
Without retrieval and gates, models invent policies/timelines. Ablation: k=0 → auto-handle rate 0. On this golden set, **TF-IDF Macro-F1 (0.683) still edges gpt-4o-mini (0.669)**—so retrieval + escalation matter more than “just use GPT for intents.”

### How do you know replies are good?
Structured rubric + **gpt-4o-mini judge** (assignment requirement). On the judge’s scoring, ResolveFlow > generic/nearest on soft quality—but human agreement is weak (correctness κ≈0.15; helpfulness κ≈−0.05; overall κ≈−0.13). Stronger agreement on hallucination/safety (κ≈0.79). Treat soft quality as **directional**, not ground truth.

### Biggest weakness?
False auto-handle among should-escalate is **4.3%** after message risk cues (was 24.3% → 9.6% → 4.3%). LLM Macro-F1 still trails TF-IDF. Legacy safety smoke **6/6**; mixed safety suite (n=36) shows **0** FAH and **2** false escalations on thin benign FAQs — not merged into the golden headline.

### Dangerous Q — “90% auto-handling?”
Auto-handle is **32%**; **safe** auto-handle **25.5%**; FAH among should-escalate **4.3%**.

### Dangerous Q — “Why believe the judge?”
Keep it: consistent automated comparisons + meaningful hallucination/safety agreement. Do **not** claim it proves one model is objectively better at helpfulness/overall. Prefer more human ratings / held-out calibration before treating soft scores as quality truth.
