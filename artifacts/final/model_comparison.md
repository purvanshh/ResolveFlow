| Metric | TF-IDF | GPT-4o-mini | DeepSeek V4.1 Flash Non-thinking | DeepSeek V4.1 Flash Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | 0.683 | 0.669 | 0.653 | 0.634 |
| Auto-handle rate | — | 0.350 | 0.380 | 0.390 |
| Safe Auto-Handling Rate | — | 0.255 | 0.255 | 0.255 |
| False Auto-handle | — | 0.096 | 0.130 | 0.139 |
| Escalation F1 | — | 0.849 | 0.837 | 0.835 |
| Reply correctness | — | 4.36 | 4.74 | 4.78 |
| Reply groundedness | — | 4.33 | 4.74 | 4.77 |
| Unsupported-claim rate | — | 0.015 | 0.020 | 0.020 |
| Safety suite | — | 6/6 | 6/6 | 6/6 |
| Average latency (s) | — | — | 1.968 | 6.980 |
| Cost | — | — | — | — |

Notes: Escalation metrics use calibrated high-risk set (`+order_quality_issue`), recomputed offline on frozen intents/replies. SAH unchanged at 0.255. Safety 6/6 after prior harness regrades.
