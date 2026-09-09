"""Dataset profiling and brand comparison helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from resolveflow.data.ingest import brand_authors, find_duplicate_tweet_ids
from resolveflow.data.threads import assign_conversation_ids


@dataclass
class BrandMetrics:
    brand: str
    tweets: int
    customer_tweets: int
    brand_replies: int
    conversations: int
    response_available: int
    usable_conversations: int
    with_2plus: int
    with_3plus: int
    customer_to_brand: int
    avg_messages: float
    median_messages: float
    max_messages: float
    suitability_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "brand": self.brand,
            "tweets": self.tweets,
            "customer_tweets": self.customer_tweets,
            "brand_replies": self.brand_replies,
            "conversations": self.conversations,
            "response_available": self.response_available,
            "usable_conversations": self.usable_conversations,
            "with_2plus": self.with_2plus,
            "with_3plus": self.with_3plus,
            "customer_to_brand": self.customer_to_brand,
            "avg_messages": self.avg_messages,
            "median_messages": self.median_messages,
            "max_messages": self.max_messages,
            "suitability_score": self.suitability_score,
        }


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a schema-level profile of the raw tweet table."""
    brands = sorted(brand_authors(df))
    dup_rows = int(df.duplicated().sum())
    dup_ids = find_duplicate_tweet_ids(df)
    null_pct = (df.isna().mean() * 100).round(3).to_dict()
    text_empty = (df["text"].fillna("").str.strip() == "").mean() * 100
    null_pct["text_empty_pct"] = round(float(text_empty), 3)

    inbound_counts = df["inbound"].value_counts(dropna=False).to_dict()
    inbound_dist = {str(k): int(v) for k, v in inbound_counts.items()}

    created = df["created_at"]
    date_min = created.min()
    date_max = created.max()

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "null_percentages": null_pct,
        "duplicate_rows": dup_rows,
        "duplicate_tweet_ids": int(len(dup_ids)),
        "unique_brands": len(brands),
        "brands": brands,
        "unique_author_ids": int(df["author_id"].nunique(dropna=True)),
        "date_range": {
            "min": None if pd.isna(date_min) else str(date_min),
            "max": None if pd.isna(date_max) else str(date_max),
        },
        "inbound_distribution": inbound_dist,
    }


def _build_conversation_index(work: pd.DataFrame) -> dict[str, Any]:
    """
    One-pass conversation summaries + brand → conversation membership.

    Returns structures used by ``compute_brand_table``.
    """
    flagged = work.assign(
        _is_customer=work["inbound"] == True,  # noqa: E712
        _is_brand=work["inbound"] == False,  # noqa: E712
        _cust_text=(work["inbound"] == True)  # noqa: E712
        & (work["text"].fillna("").str.strip() != ""),
    )
    flagged = flagged[flagged["conversation_id"].notna()].copy()
    flagged["conversation_id"] = flagged["conversation_id"].astype(int)

    g = flagged.groupby("conversation_id", sort=False)
    summary = pd.DataFrame(
        {
            "n_messages": g.size(),
            "n_customer": g["_is_customer"].sum(),
            "n_brand": g["_is_brand"].sum(),
            "n_customer_text": g["_cust_text"].sum(),
        }
    )

    brand_rows = flagged.loc[flagged["_is_brand"], ["conversation_id", "author_id"]]
    brand_to_convs: dict[str, list[int]] = defaultdict(list)
    for cid, author in zip(
        brand_rows["conversation_id"].to_numpy(),
        brand_rows["author_id"].astype(str).to_numpy(),
        strict=False,
    ):
        brand_to_convs[str(author)].append(int(cid))

    for brand, cids in list(brand_to_convs.items()):
        brand_to_convs[brand] = list(dict.fromkeys(cids))

    # Inbound tweet counts per conversation (for customer_tweets rollup)
    inbound_per_conv = (
        flagged.loc[flagged["_is_customer"]]
        .groupby("conversation_id", sort=False)
        .size()
    )

    return {
        "summary": summary,
        "brand_to_convs": brand_to_convs,
        "inbound_per_conv": inbound_per_conv,
    }


def compute_brand_table(
    df: pd.DataFrame,
    *,
    top_n: int | None = 30,
    min_brand_replies: int = 1000,
) -> list[BrandMetrics]:
    """
    Compare brands by tweet volume and reconstructed conversation quality.

    A conversation is attributed to a brand if that brand authors ≥1 outbound
    message in the reply chain. Multi-brand threads (rare) count for each.
    """
    work = df.copy()
    work["conversation_id"] = assign_conversation_ids(work)

    brand_reply_counts = (
        work.loc[work["inbound"] == False, "author_id"].value_counts()  # noqa: E712
    )
    ranked = list(brand_reply_counts.index)
    candidates = [b for b in ranked if int(brand_reply_counts[b]) >= min_brand_replies]
    if top_n is not None:
        # Keep top_n by reply volume among those passing the floor (or top_n overall)
        candidates = [b for b in ranked[: max(top_n * 2, top_n)] if b in set(candidates)]
        candidates = candidates[:top_n]
        if len(candidates) < top_n:
            # Fall back: just take top_n by replies
            candidates = list(ranked[:top_n])

    index = _build_conversation_index(work)
    summary: pd.DataFrame = index["summary"]
    brand_to_convs: dict[str, list[int]] = index["brand_to_convs"]
    inbound_per_conv: pd.Series = index["inbound_per_conv"]

    metrics: list[BrandMetrics] = []
    for brand in candidates:
        brand = str(brand)
        brand_replies = int(brand_reply_counts.get(brand, 0))
        cids = brand_to_convs.get(brand, [])

        if not cids:
            customer_tweets = 0
            stats_row = {
                "conversations": 0,
                "response_available": 0,
                "usable_conversations": 0,
                "with_2plus": 0,
                "with_3plus": 0,
                "customer_to_brand": 0,
                "avg_messages": 0.0,
                "median_messages": 0.0,
                "max_messages": 0,
            }
        else:
            # Customer tweets = inbound messages inside conversations this brand joined
            customer_tweets = int(inbound_per_conv.reindex(cids).fillna(0).sum())
            sub = summary.reindex(cids).dropna(how="all")
            n = len(sub)
            lengths = sub["n_messages"].astype(float)
            has_brand = sub["n_brand"] > 0
            has_customer = sub["n_customer"] > 0
            has_cust_text = sub["n_customer_text"] > 0
            customer_to_brand = int((has_customer & has_brand).sum())
            usable = int(
                ((lengths >= 2) & has_customer & has_brand & has_cust_text).sum()
            )
            stats_row = {
                "conversations": n,
                "response_available": int(has_brand.sum()),
                "usable_conversations": usable,
                "with_2plus": int((lengths >= 2).sum()),
                "with_3plus": int((lengths >= 3).sum()),
                "customer_to_brand": customer_to_brand,
                "avg_messages": float(lengths.mean()) if n else 0.0,
                "median_messages": float(lengths.median()) if n else 0.0,
                "max_messages": int(lengths.max()) if n else 0,
            }

        tweets = brand_replies + customer_tweets
        metrics.append(
            BrandMetrics(
                brand=brand,
                tweets=tweets,
                customer_tweets=customer_tweets,
                brand_replies=brand_replies,
                conversations=stats_row["conversations"],
                response_available=stats_row["response_available"],
                usable_conversations=stats_row["usable_conversations"],
                with_2plus=stats_row["with_2plus"],
                with_3plus=stats_row["with_3plus"],
                customer_to_brand=stats_row["customer_to_brand"],
                avg_messages=stats_row["avg_messages"],
                median_messages=stats_row["median_messages"],
                max_messages=stats_row["max_messages"],
            )
        )

    _score_brands(metrics)
    metrics.sort(key=lambda m: m.suitability_score, reverse=True)
    return metrics


def _score_brands(metrics: list[BrandMetrics]) -> None:
    """Heuristic suitability — volume + response coverage + multi-turn depth."""
    if not metrics:
        return

    usable = np.array([m.usable_conversations for m in metrics], dtype=float)
    replies = np.array([m.brand_replies for m in metrics], dtype=float)
    depth3 = np.array([m.with_3plus for m in metrics], dtype=float)
    median_len = np.array([m.median_messages for m in metrics], dtype=float)
    resp_rate = np.array(
        [
            (m.response_available / m.conversations) if m.conversations else 0.0
            for m in metrics
        ]
    )

    def norm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    scores = (
        0.35 * norm(np.log1p(usable))
        + 0.20 * norm(np.log1p(replies))
        + 0.20 * resp_rate
        + 0.15 * norm(np.log1p(depth3))
        + 0.10 * norm(np.clip(median_len, 0, 10))
    )

    for m, s in zip(metrics, scores, strict=False):
        score = float(s)
        if m.usable_conversations < 1000:
            score *= 0.4
        if m.brand_replies < 2000:
            score *= 0.5
        if m.median_messages < 2:
            score *= 0.7
        m.suitability_score = round(score, 4)


def select_brand(metrics: list[BrandMetrics]) -> BrandMetrics | None:
    if not metrics:
        return None
    return max(metrics, key=lambda m: m.suitability_score)


def format_profile_report(
    profile: dict[str, Any],
    brand_metrics: list[BrandMetrics],
    selected: BrandMetrics | None,
    *,
    top_k: int = 10,
) -> str:
    """Human-readable profile used by scripts/profile_data.py."""
    lines: list[str] = []
    lines.append("ResolveFlow Data Profile")
    lines.append("========================")
    lines.append("")
    lines.append("Dataset")
    lines.append("-------")
    lines.append(f"Rows: {profile['rows']:,}")
    dr = profile["date_range"]
    lines.append(f"Date range: {dr['min']} → {dr['max']}")
    lines.append(f"Brands: {profile['unique_brands']}")
    lines.append(f"Unique authors: {profile['unique_author_ids']:,}")
    lines.append(f"Duplicate rows: {profile['duplicate_rows']:,}")
    lines.append(f"Duplicate tweet IDs: {profile['duplicate_tweet_ids']:,}")
    lines.append("")
    lines.append("Columns")
    lines.append("-------")
    for name in profile["column_names"]:
        lines.append(f"  - {name} ({profile['dtypes'].get(name, '?')})")
    lines.append("")
    lines.append("Missing values")
    lines.append("--------------")
    for col, n in profile["missing_values"].items():
        pct = profile["null_percentages"].get(col, 0)
        lines.append(f"  {col}: {n:,} ({pct}%)")
    lines.append("")
    lines.append("Inbound distribution")
    lines.append("--------------------")
    for k, v in profile["inbound_distribution"].items():
        lines.append(f"  {k}: {v:,}")
    lines.append("")
    lines.append("Top brands")
    lines.append("----------")
    for i, m in enumerate(brand_metrics[:top_k], start=1):
        lines.append(f"{i}. {m.brand}")
        lines.append(f"   Tweets: {m.tweets:,}")
        lines.append(f"   Customer: {m.customer_tweets:,}")
        lines.append(f"   Brand: {m.brand_replies:,}")
        lines.append(f"   Conversations: {m.conversations:,}")
        lines.append(f"   Usable: {m.usable_conversations:,}")
        lines.append(f"   Response available: {m.response_available:,}")
        lines.append(f"   Median length: {m.median_messages:.1f}")
        lines.append(f"   Suitability: {m.suitability_score:.4f}")
        lines.append("")

    lines.append("Selected brand")
    lines.append("--------------")
    if selected is None:
        lines.append("(none)")
    else:
        lines.append(selected.brand)
        lines.append("")
        lines.append("Usable conversations")
        lines.append("--------------------")
        lines.append(f"{selected.usable_conversations:,}")
        lines.append("")
        lines.append("Response available")
        lines.append("------------------")
        lines.append(f"{selected.response_available:,}")
        lines.append("")
        lines.append("Median conversation length")
        lines.append("--------------------------")
        lines.append(f"{selected.median_messages:.1f}")
        lines.append("")
        lines.append("Avg / max conversation length")
        lines.append("-----------------------------")
        lines.append(f"{selected.avg_messages:.2f} / {int(selected.max_messages)}")
    lines.append("")
    return "\n".join(lines)


def brand_comparison_markdown(metrics: list[BrandMetrics], *, limit: int = 20) -> str:
    header = (
        "| Brand | Tweets | Customer Tweets | Brand Replies | Conversations |"
        " Usable | Response avail. | Median msgs | Score |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for m in metrics[:limit]:
        rows.append(
            f"| {m.brand} | {m.tweets:,} | {m.customer_tweets:,} | {m.brand_replies:,} | "
            f"{m.conversations:,} | {m.usable_conversations:,} | {m.response_available:,} | "
            f"{m.median_messages:.1f} | {m.suitability_score:.4f} |"
        )
    return header + "\n" + "\n".join(rows)
