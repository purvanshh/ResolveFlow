| Metric | TF-IDF | GPT-4o-mini | DeepSeek V4.1 Flash Non-thinking | DeepSeek V4.1 Flash Thinking |
| --- | ---: | ---: | ---: | ---: |
| Intent Macro-F1 | 0.683 | 0.669 | 0.653 | 0.634 |
| Auto-handle rate | — | 0.440 | 0.470 | 0.500 |
| Safe Auto-Handling Rate | — | 0.255 | 0.255 | 0.255 |
| False Auto-handle | — | 0.243 | 0.270 | 0.322 |
| Escalation F1 | — | 0.767 | 0.760 | 0.726 |
| Reply correctness | — | 4.36 | 4.74 | 4.78 |
| Reply groundedness | — | 4.33 | 4.74 | 4.77 |
| Unsupported-claim rate | — | 0.015 | 0.020 | 0.020 |
| Safety suite | — | 5/6 | 6/6 | 6/6 |
| Average latency (s) | — | — | 1.968 | 6.980 |
| Cost | — | — | — | — |

Notes: GPT 5/6 fail is `account_specific`. DeepSeek 6/6 is after negation-aware regrade of stored `fake_policy` replies (was harness false positive on safe negation; model text unchanged).
