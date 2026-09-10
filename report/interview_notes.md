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
Without retrieval and gates, models invent policies/timelines. Ablation: k=0 → auto-handle rate 0 (policy requires evidence). Retrieval doesn’t raise intent F1 here; it enables grounded drafting and automation gates.

### Why escalation?
No account APIs. High-risk intents and weak evidence must go to humans. Confidence-only FAH 0.383 vs proposed 0.165 (among should-escalate).

### How do you know replies are good?
Structured rubric (correctness, groundedness, helpfulness, completeness, tone, hallucination safety, escalation). Heuristic judge cached; 40 solo human ratings; Streamlit UI for independent rating. Compare generic + nearest-case baselines. **Do not** cite a single LLM score as proof.

### Biggest weakness?
False auto-handle still 16.5% of should-escalate cases; offline agent doesn’t show LLM intent gains; judge/human agreement variance is too low to claim strong calibration.

### What is misleading about the headline?
n=200, stratified Twitter sample, one brand, offline templates≠GPT, no CSAT, judge saturation, solo calibrator — see report §9.

### What next?
Hybrid retrieval + FAH-constrained policy tuning + OpenAI eval under frozen prompts + second annotator.

### Dangerous Q — “90% auto-handling?”
We don’t claim that. Auto-handle is **26%**; **safe** auto-handle **16.5%**; FAH among should-escalate **16.5%**. Automation without FAH is not trustworthy.

### Dangerous Q — “Did retrieval memorize the test set?”
Golden conversations excluded from the corpus; leakage checks for IDs and text are zero.

### Dangerous Q — “Why believe the judge?”
We don’t fully. Heuristic judge + solo human calibration; dimensions with constant 5s yield Spearman 0. Treat as noisy diagnostic until independent Streamlit ratings exist.
