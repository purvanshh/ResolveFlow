"""Build historical support cases from development customer messages."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

from resolveflow.annotate import annotate_example
from resolveflow.data.normalize import normalize_customer_text


def _english_ish(text: str) -> bool:
    t = str(text or "")
    letters = re.findall(r"[A-Za-z]", t)
    non = re.findall(r"[^\x00-\x7F]", t)
    if len(letters) < 12:
        return len(non) == 0 and len(t.strip()) >= 12
    return (len(letters) / max(len(t), 1)) > 0.55 and len(non) < len(letters) * 0.35


def is_usable_case(row: pd.Series) -> bool:
    cust = str(row.get("raw_text") or "").strip()
    resp = str(row.get("brand_response") or "").strip()
    if len(cust) < 12 or len(resp) < 12:
        return False
    if cust.lower() in {"[url]", "ok", "thanks", "thank you", "yes", "no"}:
        return False
    # Brand reply should look like a reply, not empty boilerplate alone without content
    if resp.lower() in {"", "nan", "none"}:
        return False
    return True


def deterministic_resolution_summary(customer: str, brand_response: str) -> str:
    """
    Compact, non-inventive summary from the historical brand response only.

    Does not invent refunds, timelines, or outcomes absent from the reply text.
    """
    resp = re.sub(r"\s+", " ", str(brand_response)).strip()
    # Strip handles for readability
    resp = re.sub(r"@\w+", "", resp).strip()
    resp = re.sub(r"https?://\S+", "[URL]", resp)
    if len(resp) > 220:
        resp = resp[:217].rstrip() + "..."
    return f"Brand replied: {resp}"


def build_historical_cases(
    messages: pd.DataFrame,
    *,
    brand: str = "AmazonHelp",
    max_cases: int | None = 20000,
    seed: int = 42,
    prefer_english: bool = True,
    label_intents: bool = True,
) -> pd.DataFrame:
    """
    Convert development customer messages into retrieval cases.

    Requires a non-empty brand_response. Optionally silver-labels intent via
    the Phase-2 annotation rules (not golden labels).
    """
    work = messages.copy()
    work = work[work.apply(is_usable_case, axis=1)]
    if prefer_english:
        work = work[work["raw_text"].map(_english_ish)]

    # Prefer cases with some context or longer customer text
    work = work.assign(
        _score=work["raw_text"].astype(str).str.len()
        + work["customer_context"].fillna("").astype(str).str.len().clip(upper=400) * 0.25
        + (work["n_messages"].fillna(1).astype(float) >= 3).astype(int) * 20
    )
    work = work.sort_values("_score", ascending=False)

    # At most one case per conversation (first/highest score customer turn with reply)
    work = work.drop_duplicates(subset=["conversation_id"], keep="first")

    if max_cases is not None and len(work) > max_cases:
        # Stratify a bit: keep top by score within random sample mix
        top = work.head(int(max_cases * 0.7))
        rest = work.iloc[int(max_cases * 0.7) :]
        extra = rest.sample(
            n=min(len(rest), max_cases - len(top)), random_state=seed
        )
        work = pd.concat([top, extra], ignore_index=True)

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        cust = str(row["raw_text"])
        ctx = "" if pd.isna(row.get("customer_context")) else str(row["customer_context"])
        resp = str(row["brand_response"])
        intent = ""
        if label_intents:
            lab = annotate_example(cust, ctx)
            intent = lab.intent
        case_id = hashlib.sha1(
            f"{row['conversation_id']}:{row['message_id']}".encode()
        ).hexdigest()[:16]
        rows.append(
            {
                "case_id": case_id,
                "conversation_id": str(int(row["conversation_id"])),
                "message_id": str(int(row["message_id"])),
                "customer_message": cust,
                "customer_context": ctx,
                "brand_response": resp,
                "resolution_summary": deterministic_resolution_summary(cust, resp),
                "intent": intent,
                "timestamp": row.get("timestamp"),
                "num_messages": int(row.get("n_messages") or 1),
                "normalized_query": normalize_customer_text(cust, brand=brand),
            }
        )
    return pd.DataFrame(rows)


def cases_checksum(df: pd.DataFrame) -> str:
    payload = (
        df[["case_id", "conversation_id", "message_id"]]
        .astype(str)
        .sort_values("case_id")
        .to_csv(index=False)
        .encode()
    )
    return hashlib.sha256(payload).hexdigest()


def exclude_golden_from_messages(
    messages: pd.DataFrame, golden: pd.DataFrame
) -> pd.DataFrame:
    """Defense-in-depth exclusion (dev parquet should already be clean)."""
    g_convs = set(golden["conversation_id"].astype(str))
    g_msgs = set(golden["message_id"].astype(str))
    g_text = set(golden["input_text"].astype(str).str.strip())
    mask = (
        ~messages["conversation_id"].astype(str).isin(g_convs)
        & ~messages["message_id"].astype(str).isin(g_msgs)
        & ~messages["raw_text"].astype(str).str.strip().isin(g_text)
    )
    return messages.loc[mask].copy()
