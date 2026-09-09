"""Build brand-scoped customer-message tables for intent discovery."""

from __future__ import annotations

from typing import Any

import pandas as pd

from resolveflow.data.normalize import normalize_customer_text
from resolveflow.data.threads import assign_conversation_ids


def build_brand_customer_messages(
    df: pd.DataFrame,
    brand: str,
) -> pd.DataFrame:
    """
    One row per customer message in conversations that include ``brand``.

    Columns:
      conversation_id, message_id, timestamp, raw_text, normalized_text,
      customer_context, brand_response, n_messages
    """
    work = df.copy()
    if "conversation_id" not in work.columns:
        work["conversation_id"] = assign_conversation_ids(work)

    brand_cids = set(
        work.loc[
            (work["author_id"] == brand) & (work["inbound"] == False),  # noqa: E712
            "conversation_id",
        ]
        .dropna()
        .astype(int)
    )
    scoped = work[work["conversation_id"].isin(brand_cids)].copy()
    scoped = scoped[scoped["conversation_id"].notna()]
    scoped["conversation_id"] = scoped["conversation_id"].astype(int)

    rows: list[dict[str, Any]] = []
    for cid, group in scoped.groupby("conversation_id", sort=False):
        ordered = group.sort_values(
            by=["created_at", "tweet_id"], ascending=True, na_position="last"
        )
        messages = ordered.to_dict("records")
        n_messages = len(messages)
        for i, msg in enumerate(messages):
            if msg.get("inbound") is not True:
                continue
            raw = "" if pd.isna(msg.get("text")) else str(msg["text"])
            # Prior messages as short context
            prior = messages[:i]
            context_bits = []
            for p in prior[-4:]:
                role = "customer" if p.get("inbound") is True else "brand"
                t = "" if pd.isna(p.get("text")) else str(p["text"])
                context_bits.append(f"{role}: {t}")
            # Next brand reply after this customer message (if any)
            brand_response = ""
            for later in messages[i + 1 :]:
                if later.get("inbound") is False and str(later.get("author_id")) == brand:
                    brand_response = (
                        "" if pd.isna(later.get("text")) else str(later["text"])
                    )
                    break
            ts = msg.get("created_at")
            rows.append(
                {
                    "conversation_id": int(cid),
                    "message_id": (
                        None if pd.isna(msg.get("tweet_id")) else int(msg["tweet_id"])
                    ),
                    "timestamp": None if pd.isna(ts) else pd.Timestamp(ts).isoformat(),
                    "raw_text": raw,
                    "normalized_text": normalize_customer_text(raw, brand=brand),
                    "customer_context": "\n".join(context_bits),
                    "brand_response": brand_response,
                    "n_messages": n_messages,
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out = out[out["normalized_text"].str.strip() != ""].reset_index(drop=True)
    return out


def sample_intent_discovery(
    messages: pd.DataFrame,
    *,
    n: int = 4000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Representative mix: random + longer threads + short msgs + time strata.
    """
    if len(messages) <= n:
        return messages.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = seed
    parts: list[pd.DataFrame] = []
    remaining = messages.copy()

    # 40% pure random
    n_rand = int(n * 0.40)
    take = remaining.sample(n=min(n_rand, len(remaining)), random_state=rng)
    parts.append(take)
    remaining = remaining.drop(take.index)

    # 20% longer conversations
    n_long = int(n * 0.20)
    long_pool = remaining[remaining["n_messages"] >= 4]
    if len(long_pool):
        take = long_pool.sample(n=min(n_long, len(long_pool)), random_state=rng + 1)
        parts.append(take)
        remaining = remaining.drop(take.index)

    # 15% short messages
    n_short = int(n * 0.15)
    short_pool = remaining[remaining["normalized_text"].str.len() <= 40]
    if len(short_pool):
        take = short_pool.sample(n=min(n_short, len(short_pool)), random_state=rng + 2)
        parts.append(take)
        remaining = remaining.drop(take.index)

    # 25% stratified by time quartile if timestamps exist
    n_time = n - sum(len(p) for p in parts)
    if n_time > 0 and len(remaining):
        tmp = remaining.copy()
        tmp["_ts"] = pd.to_datetime(tmp["timestamp"], errors="coerce", utc=True)
        if tmp["_ts"].notna().any():
            tmp["_q"] = pd.qcut(tmp["_ts"].rank(method="first"), 4, labels=False)
            per = max(1, n_time // 4)
            for q in range(4):
                pool = tmp[tmp["_q"] == q]
                if not len(pool):
                    continue
                take = pool.sample(n=min(per, len(pool)), random_state=rng + 10 + q)
                parts.append(take.drop(columns=["_ts", "_q"], errors="ignore"))
                remaining = remaining.drop(take.index)
        # top up from remaining
        need = n - sum(len(p) for p in parts)
        if need > 0 and len(remaining):
            parts.append(
                remaining.sample(n=min(need, len(remaining)), random_state=rng + 99)
            )

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["message_id"], keep="first")
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)
