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
Structured rubric + **gpt-4o-mini judge** vs generic/nearest baselines (correctness 4.36 vs 3.24/3.46). Human calibration n=40: correctness Spearman ≈ 0.30—treat judge as noisy.

### Biggest weakness?
False auto-handle among should-escalate is **9.6%** after escalation calibration (`order_quality_issue` high-risk); was 24.3%. LLM does not beat TF-IDF on intent; safety suite **6/6**.

### Dangerous Q — “90% auto-handling?”
Auto-handle is **35%**; **safe** auto-handle **25.5%**; FAH among should-escalate **9.6%** (was 24.3% before calibrating `order_quality_issue` as high-risk).

### Dangerous Q — “Why believe the judge?”
gpt-4o-mini judge with solo human calibration; modest correctness correlation (~0.30). Prefer Streamlit independent ratings before trusting reply scores alone.
