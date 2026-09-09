# Brand Selection

## Dataset

Source: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(`thoughtvector/customer-support-on-twitter`).

The working copy used for Phase 1 is `data/raw/twcs.csv` (≈493 MB). Kaggle API
credentials were not available in this environment, so the file was obtained from a
public Hugging Face mirror of the same CSV (`SunidhiSriram/twcs`). Schema and row
counts match the Kaggle dataset description.

| Field | Observed value |
| --- | --- |
| Rows | 2,811,774 |
| Columns | `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`, `in_response_to_tweet_id` |
| Brands (outbound authors) | 108 |
| Unique authors | 702,777 |
| Date range | 2008-05-08 → 2017-12-03 (bulk activity is much more recent) |
| Inbound (customer) | 1,537,843 |
| Outbound (brand) | 1,273,931 |
| Duplicate rows / tweet IDs | 0 / 0 |
| Missing `response_tweet_id` | 37.0% |
| Missing `in_response_to_tweet_id` | 28.3% (roots of reply chains) |

### Conversation reconstruction

We treat each reply chain as a conversation:

1. Follow `in_response_to_tweet_id` until a tweet with no parent (or a cycle break).
2. Use that **root `tweet_id` as the deterministic `conversation_id`**.
3. Order messages by `created_at`, then `tweet_id`.
4. Label roles with `inbound` (`True` → customer, `False` → brand).
5. Attribute a conversation to a brand if that brand authors ≥1 outbound tweet in the chain.

Normalized message shape:

```python
{
    "conversation_id": "...",
    "brand": "...",
    "messages": [
        {"tweet_id": "...", "role": "customer"|"brand", "text": "...", "timestamp": "..."}
    ]
}
```

**Important proxy language:** the Kaggle corpus is already filtered so included
threads generally contain a consumer request and a company reply. Within
brand-attributed conversations, `response_available` is therefore near-tautological.
We still report it, but we do **not** treat it as evidence of successful resolution.

`usable_conversations` = conversations with ≥2 messages, ≥1 customer message with
non-empty text, and ≥1 brand message.

## Candidate Brands

Top brands by Phase 1 suitability score (reply-chain stats):

| Brand | Tweets | Customer Tweets | Brand Replies | Conversations | Usable | Response avail. | Median msgs | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AmazonHelp | 373,438 | 203,598 | 169,840 | 82,534 | 82,534 | 82,534 | 3.0 | 0.9500 |
| AppleSupport | 238,624 | 131,764 | 106,860 | 80,702 | 80,702 | 80,702 | 2.0 | 0.8205 |
| Uber_Support | 128,424 | 72,154 | 56,270 | 41,923 | 41,923 | 41,923 | 2.0 | 0.6273 |
| Tesco | 72,801 | 34,228 | 38,573 | 16,721 | 16,721 | 16,721 | 4.0 | 0.5382 |
| SpotifyCares | 91,808 | 48,543 | 43,265 | 28,280 | 28,277 | 28,280 | 2.0 | 0.5224 |
| Delta | 87,549 | 45,296 | 42,253 | 26,166 | 26,164 | 26,166 | 2.0 | 0.5121 |
| AmericanAir | 86,818 | 50,054 | 36,764 | 26,385 | 26,385 | 26,385 | 2.0 | 0.5039 |
| comcastcares | 72,610 | 39,579 | 33,031 | 24,061 | 24,061 | 24,061 | 2.0 | 0.4638 |
| TMobileHelp | 81,475 | 47,158 | 34,317 | 22,789 | 22,789 | 22,789 | 2.0 | 0.4575 |
| British_Airways | 60,548 | 31,187 | 29,361 | 16,450 | 16,450 | 16,450 | 3.0 | 0.4457 |

Tweet totals above = brand outbound replies + inbound tweets inside conversations
that brand joined (not a global `@mention` scrape).

## Selection Criteria

Heuristic suitability (documented in `src/resolveflow/data/profile.py`), emphasizing:

1. **Sufficient usable volume** (log-scaled usable conversations)
2. **Sufficient brand responses** (historical reply inventory for retrieval later)
3. **Response coverage** among attributed conversations
4. **Multi-turn depth** (3+ message conversations; median length)
5. Soft penalties for tiny brands / median length &lt; 2

Qualitative checks (sampled first customer messages):

| Brand | Recurring themes (rough lexical signal) |
| --- | --- |
| AmazonHelp | delivery, order, prime, customer service, shipping |
| AppleSupport | iPhone, iOS, update, battery, “please fix” |
| Uber_Support | driver, ride, charged, account, app |
| Tesco | store, delivery, click & collect, out-of-date |
| SpotifyCares | premium, account, app, songs / playback |

We preferred a brand with **thousands of usable multi-turn threads**, repeated
support patterns, and enough lexical diversity to support a multi-intent taxonomy
in Phase 2 — without choosing the absolute largest brand blindly if depth were weak.

## Selected Brand

**AmazonHelp**

## Why?

1. **Volume with depth:** ~82.5k usable conversations; median length **3.0** messages
   (vs AppleSupport’s median **2.0**), average **4.53**, p90 ≈ 8. Enough multi-turn
   material for intent + escalation work without relying on single ping-pong pairs.
2. **Historical brand replies:** ~169.8k outbound tweets — a large retrieval pool for
   later reply / style matching.
3. **Repeated but multi-theme support patterns:** delivery / order / Prime / customer
   service dominate, but they are distinct operational intents (shipping delay vs
   order wrong vs membership vs general service complaint), not a single canned issue.
4. **Manageable relative to alternatives:** AppleSupport is a close second and is more
   device/software-centric; Uber is more payment/driver-centric with shorter median
   threads. Amazon struck the better balance of scale, reply inventory, and turn depth
   for a single-brand Phase 2 taxonomy.

We selected AmazonHelp because it had sufficient conversation volume and a relatively
diverse set of recurring support issues. We did **not** claim that these conversations
represent successful resolutions; we use the presence of a brand response as a proxy
for historical support behavior (`response_available`).

## Conversation statistics (AmazonHelp)

| Metric | Value |
| --- | ---: |
| Total tweets (brand-attributed) | 373,438 |
| Customer tweets | 203,598 |
| Brand replies | 169,840 |
| Conversations | 82,534 |
| Usable conversations | 82,534 |
| Response available | 82,534 |
| Conversations with 2+ messages | (≈ all usable; corpus construction) |
| Conversations with 3+ messages | majority (median = 3) |
| Avg messages / conversation | 4.53 |
| Median messages / conversation | 3.0 |
| Max messages / conversation | 448 |

Length note: ~38% of AmazonHelp conversations are exactly 2 messages; ~7% have ≥10.
The max of 448 is an extreme outlier (likely a long-running or oddly linked thread)
and should be capped or filtered before modeling.

## Limitations

- **Not resolution labels.** Brand reply ≠ issue fixed. Many replies are redirects to
  DM / forms / links.
- **Attribution bias.** Conversations without any brand outbound tweet are not counted
  under that brand; orphan customer shouts into the void are underrepresented.
- **Amazon topical skew.** Logistics/order themes dominate. Software-bug or streaming
  intents will be scarce compared with AppleSupport / SpotifyCares.
- **Noise.** Mentions, sarcasm, multi-party threads, and DM redirects remain.
- **Language / geo mix.** English-heavy but not exclusively; no language filter yet.
- **Dataset acquisition path.** Used an HF mirror because Kaggle credentials were
  unavailable; file identity should be re-verified against Kaggle if credentials appear.
- **Date span.** Absolute min date (2008) suggests sparse early rows; analysis should
  prefer recent mass of traffic when splitting train/eval later.

## Reproducibility

```bash
pip install -e ".[dev]"
python scripts/profile_data.py --write-interim
pytest
```

Artifacts:

- `configs/brand.yaml` — selected brand
- `data/interim/phase1_brand_metrics.json` — full comparison metrics
- `data/interim/phase1_profile.txt` — console profile dump
