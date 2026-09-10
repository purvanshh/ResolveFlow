"""Rank and summarize agent failures for inspection."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def failure_score(row: dict[str, Any]) -> float:
    """
    Practical prioritization score (not a universal scientific formula).

    severity weights favor safety + false auto-handle.
    """
    score = 0.0
    flags = row.get("safety_flags") or []
    if flags:
        score += 3.0
    gold_esc = row.get("gold_escalation")
    pred_esc = row.get("predicted_escalation")
    if gold_esc == "escalate" and pred_esc == "auto_handle":
        score += 3.0  # false auto-handle
    if not row.get("intent_correct", True):
        score += 2.0
    if not row.get("escalation_correct", True):
        score += 1.0
    # low reply quality if judge overall present
    overall = row.get("judge_overall")
    if overall is not None and float(overall) <= 2:
        score += 1.0
    if (row.get("reply") or "") and any(
        x in (row.get("reply") or "").lower()
        for x in ["we have refunded", "within 3 days", "$"]
    ):
        score += 1.0
    return score


def rank_failures(rows: list[dict[str, Any]], top_n: int = 50) -> list[dict[str, Any]]:
    scored = []
    for r in rows:
        s = failure_score(r)
        if s <= 0 and r.get("intent_correct") and r.get("escalation_correct"):
            continue
        item = dict(r)
        item["failure_score"] = s
        scored.append(item)
    scored.sort(key=lambda x: x["failure_score"], reverse=True)
    return scored[:top_n]


def categorize_failure(row: dict[str, Any]) -> str:
    gold_esc = row.get("gold_escalation")
    pred_esc = row.get("predicted_escalation")
    if gold_esc == "escalate" and pred_esc == "auto_handle":
        return "false_auto_handle"
    if row.get("safety_flags"):
        return "safety_violation"
    if not row.get("intent_correct"):
        gi = row.get("gold_intent")
        pi = row.get("predicted_intent")
        if {gi, pi} == {"refund_request", "refund_status"}:
            return "intent_confusion_refund"
        if {gi, pi} == {"delivery_delay", "package_missing_or_misdelivered"}:
            return "intent_confusion_delivery"
        if gi == "other_unclear" or pi == "other_unclear":
            return "ambiguous_or_unclear"
        return "intent_error_other"
    if gold_esc == "auto_handle" and pred_esc == "escalate":
        return "over_escalation"
    text = (row.get("customer_message") or row.get("input_text") or "")
    if len(str(text).strip()) <= 25:
        return "ambiguous_short_message"
    return "other"


def summarize_failure_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cats = Counter(categorize_failure(r) for r in rows if failure_score(r) > 0 or not r.get("intent_correct") or not r.get("escalation_correct"))
    severity = {
        "false_auto_handle": "Critical",
        "safety_violation": "High",
        "intent_confusion_refund": "Medium",
        "intent_confusion_delivery": "Medium",
        "ambiguous_or_unclear": "High",
        "ambiguous_short_message": "High",
        "intent_error_other": "Medium",
        "over_escalation": "Low",
        "other": "Low",
    }
    root = {
        "false_auto_handle": "Risk policy / annotator conservatism mismatch on low-risk intents",
        "safety_violation": "Generation overconfidence or weak safety gate",
        "intent_confusion_refund": "Taxonomy boundary overlap (request vs status)",
        "intent_confusion_delivery": "Topic-level lexical overlap (late vs missing)",
        "ambiguous_or_unclear": "Missing context; forced classification risk",
        "ambiguous_short_message": "Insufficient information in tweet",
        "intent_error_other": "Classifier lexical confusion",
        "over_escalation": "Conservative policy (often acceptable)",
        "other": "Mixed",
    }
    fix = {
        "false_auto_handle": "Tighten auto-handle eligibility; calibrate to human escalate labels on validation",
        "safety_violation": "Expand deterministic claim detectors; block auto-handle on any flag",
        "intent_confusion_refund": "Strengthen annotation boundaries; add disambiguation features",
        "intent_confusion_delivery": "Hybrid retrieval + delivery-status features",
        "ambiguous_or_unclear": "Context builder + abstain policy (already partial)",
        "ambiguous_short_message": "Require clarifying question / escalate",
        "intent_error_other": "More labeled train data; LLM classifier when API available",
        "over_escalation": "Acceptable tradeoff; optional soft auto-handle for clear delivery_delay",
        "other": "Manual review",
    }
    out = []
    for mode, freq in cats.most_common():
        out.append(
            {
                "failure_mode": mode,
                "frequency": int(freq),
                "severity": severity.get(mode, "Medium"),
                "root_cause": root.get(mode, ""),
                "proposed_fix": fix.get(mode, ""),
            }
        )
    return out


def confusion_pairs(rows: list[dict[str, Any]], top_n: int = 8) -> list[tuple[str, str, int]]:
    c: Counter[tuple[str, str]] = Counter()
    for r in rows:
        if r.get("gold_intent") != r.get("predicted_intent"):
            c[(r.get("gold_intent"), r.get("predicted_intent"))] += 1
    return [(a, b, n) for (a, b), n in c.most_common(top_n)]
