| Metric | TF-IDF | GPT-4o-mini | DeepSeek V4.1 Flash Non-thinking | DeepSeek V4.1 Flash Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | 0.683 | 0.669 | 0.653 | 0.634 |
| Auto-handle rate | — | 0.320 | 0.355 | 0.365 |
| Safe Auto-Handling Rate | — | 0.255 | 0.255 | 0.255 |
| False Auto-handle | — | 0.043 | 0.087 | 0.096 |
| Escalation F1 | — | 0.876 | 0.861 | 0.860 |
| Reply correctness | — | 4.36 | 4.74 | 4.78 |
| Reply groundedness | — | 4.33 | 4.74 | 4.77 |
| Unsupported-claim rate | — | 0.015 | 0.020 | 0.020 |
| Safety suite | — | 6/6 | 6/6 | 6/6 |
| Average latency (s) | — | — | 1.968 | 6.980 |
| Cost | — | — | — | — |

Notes: Message-level risk cues adopted on top of `order_quality_issue` high-risk. SAH=0.255 unchanged. Hybrid GPT+TF-IDF not adopted.
