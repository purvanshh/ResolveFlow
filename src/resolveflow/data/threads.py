"""Conversation / thread reconstruction from tweet reply links."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from resolveflow.data.ingest import brand_authors, parse_response_ids


Role = str  # "customer" | "brand" | "unknown"


def _find_roots(
    tweet_ids: np.ndarray,
    parents: np.ndarray,
) -> dict[int, int]:
    """
    Map each tweet_id -> root conversation_id by walking in_response_to links.

    Missing parents are treated as roots. Cycles are broken by stopping when a
    visited node is re-seen (malformed rows must not crash the pipeline).
    """
    parent_of: dict[int, int] = {}
    for tid, parent in zip(tweet_ids, parents, strict=False):
        if pd.isna(tid):
            continue
        tid_i = int(tid)
        if pd.isna(parent):
            continue
        parent_of[tid_i] = int(parent)

    root_cache: dict[int, int] = {}

    def root_of(tid: int) -> int:
        if tid in root_cache:
            return root_cache[tid]
        seen: list[int] = []
        cur = tid
        while True:
            if cur in root_cache:
                root = root_cache[cur]
                break
            if cur in seen:
                # Cycle — treat earliest visited as root.
                root = seen[0]
                break
            seen.append(cur)
            parent = parent_of.get(cur)
            if parent is None:
                root = cur
                break
            cur = parent
        for node in seen:
            root_cache[node] = root
        return root

    for tid in tweet_ids:
        if pd.isna(tid):
            continue
        root_of(int(tid))
    return root_cache


def assign_conversation_ids(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic conversation IDs: the root tweet_id of each reply chain.

    Returns a Series aligned to ``df.index``.
    """
    roots = _find_roots(
        df["tweet_id"].to_numpy(),
        df["in_response_to_tweet_id"].to_numpy(),
    )
    conv_ids = []
    for tid in df["tweet_id"]:
        if pd.isna(tid):
            conv_ids.append(pd.NA)
        else:
            conv_ids.append(roots.get(int(tid), int(tid)))
    return pd.Series(conv_ids, index=df.index, dtype="Int64")


def message_role(inbound: object, author_id: str, brands: set[str]) -> Role:
    if inbound is True or (isinstance(inbound, (int, np.integer)) and inbound == 1):
        return "customer"
    if inbound is False or (isinstance(inbound, (int, np.integer)) and inbound == 0):
        return "brand"
    if str(author_id) in brands:
        return "brand"
    if str(author_id).isdigit():
        return "customer"
    return "unknown"


def reconstruct_conversations(
    df: pd.DataFrame,
    *,
    brand: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build normalized conversation dicts from a tweet DataFrame.

    If ``brand`` is set, only conversations that include that brand author
    (outbound) are returned.
    """
    work = df.copy()
    if "conversation_id" not in work.columns:
        work["conversation_id"] = assign_conversation_ids(work)

    brands = brand_authors(work)
    if brand is not None:
        brand = str(brand)
        brand_tweet_ids = set(
            work.loc[work["author_id"] == brand, "conversation_id"].dropna().astype(int)
        )
        work = work[work["conversation_id"].isin(brand_tweet_ids)]

    conversations: list[dict[str, Any]] = []
    for conv_id, group in work.groupby("conversation_id", sort=False):
        if pd.isna(conv_id):
            continue
        ordered = group.sort_values(
            by=["created_at", "tweet_id"],
            ascending=True,
            na_position="last",
        )
        messages = []
        brand_in_thread: list[str] = []
        for _, row in ordered.iterrows():
            role = message_role(row["inbound"], row["author_id"], brands)
            if role == "brand":
                brand_in_thread.append(str(row["author_id"]))
            ts = row["created_at"]
            messages.append(
                {
                    "tweet_id": None if pd.isna(row["tweet_id"]) else int(row["tweet_id"]),
                    "author_id": str(row["author_id"]),
                    "role": role,
                    "text": "" if pd.isna(row["text"]) else str(row["text"]),
                    "timestamp": None if pd.isna(ts) else pd.Timestamp(ts).isoformat(),
                }
            )
        primary_brand = _majority_brand(brand_in_thread)
        conversations.append(
            {
                "conversation_id": str(int(conv_id)),
                "brand": primary_brand,
                "messages": messages,
            }
        )
    return conversations


def _majority_brand(brand_ids: list[str]) -> str | None:
    if not brand_ids:
        return None
    counts: dict[str, int] = defaultdict(int)
    for b in brand_ids:
        counts[b] += 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def conversation_stats(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate volume / quality / length stats for a conversation list."""
    n = len(conversations)
    if n == 0:
        return {
            "conversations": 0,
            "with_brand_reply": 0,
            "with_2plus_messages": 0,
            "with_3plus_messages": 0,
            "customer_to_brand": 0,
            "response_available": 0,
            "avg_messages": 0.0,
            "median_messages": 0.0,
            "max_messages": 0,
            "usable_conversations": 0,
        }

    lengths: list[int] = []
    with_brand = 0
    with_2 = 0
    with_3 = 0
    customer_to_brand = 0
    usable = 0

    for conv in conversations:
        msgs = conv["messages"]
        lengths.append(len(msgs))
        roles = [m["role"] for m in msgs]
        has_customer = "customer" in roles
        has_brand = "brand" in roles
        if has_brand:
            with_brand += 1
        if len(msgs) >= 2:
            with_2 += 1
        if len(msgs) >= 3:
            with_3 += 1
        if has_customer and has_brand:
            customer_to_brand += 1
            # Usable: at least one customer message + one brand reply, 2+ msgs,
            # and at least one non-empty customer text.
            customer_texts = [
                m["text"].strip() for m in msgs if m["role"] == "customer"
            ]
            if len(msgs) >= 2 and any(customer_texts):
                usable += 1

    arr = np.asarray(lengths, dtype=float)
    return {
        "conversations": n,
        "with_brand_reply": with_brand,
        "with_2plus_messages": with_2,
        "with_3plus_messages": with_3,
        "customer_to_brand": customer_to_brand,
        "response_available": with_brand,
        "avg_messages": float(arr.mean()),
        "median_messages": float(np.median(arr)),
        "max_messages": int(arr.max()),
        "usable_conversations": usable,
    }


def expand_response_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Optional helper: explode response_tweet_id lists into edge rows."""
    rows = []
    for _, row in df.iterrows():
        for rid in parse_response_ids(row.get("response_tweet_id")):
            rows.append(
                {
                    "tweet_id": row["tweet_id"],
                    "response_tweet_id": rid,
                }
            )
    return pd.DataFrame(rows)
