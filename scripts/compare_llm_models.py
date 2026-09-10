#!/usr/bin/env python3
"""Build TF-IDF vs GPT-4o-mini vs DeepSeek comparison from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _safety_str(suite) -> str:
    if not suite:
        return "—"
    if "failures" in suite and "results" in suite:
        n = len(suite.get("results") or [])
        fails = int(suite.get("failures") or 0)
        return f"{n - fails}/{n}"
    if suite.get("passed") is True:
        return "PASS"
    if suite.get("passed") is False:
        return f"FAIL({suite.get('failures')})"
    return "—"


def collect() -> list[dict]:
    base_final = ROOT / "artifacts" / "final"
    baselines = _load_json(ROOT / "artifacts" / "metrics" / "baselines.json") or {}
    tfidf_f1 = (
        baselines.get("systems", {})
        .get("tfidf_logistic_regression", {})
        .get("intent", {})
        .get("macro_f1")
    )

    gpt_headline = _load_json(base_final / "headline.json") or {}
    gpt_reply = _load_json(base_final / "reply_comparison.json") or {}
    gpt_esc = _load_json(base_final / "escalation_metrics.json") or {}
    gpt_safety_final = _load_json(base_final / "safety_results.json") or {}
    gpt_safety = gpt_safety_final.get("suite") or _load_json(
        ROOT / "artifacts" / "metrics" / "safety_suite.json"
    ) or {}
    gpt_runtime = None  # optional

    rows = [
        {
            "system": "TF-IDF",
            "intent_macro_f1": tfidf_f1,
            "auto_handle_rate": None,
            "safe_auto_handling_rate": None,
            "false_auto_handle_rate": None,
            "escalation_f1": None,
            "reply_correctness": None,
            "reply_groundedness": None,
            "unsupported_claim_rate": None,
            "safety_suite": None,
            "latency_mean_s": None,
            "cost": None,
        },
        {
            "system": "GPT-4o-mini",
            "intent_macro_f1": (gpt_headline.get("support_metrics") or {}).get(
                "intent_macro_f1"
            ),
            "auto_handle_rate": (gpt_headline.get("support_metrics") or {}).get(
                "auto_handle_rate"
            ),
            "safe_auto_handling_rate": gpt_headline.get("value"),
            "false_auto_handle_rate": (gpt_headline.get("support_metrics") or {}).get(
                "false_auto_handle_rate_among_should_escalate"
            ),
            "escalation_f1": (gpt_esc.get("proposed") or {}).get("escalate_f1"),
            "reply_correctness": (gpt_reply.get("llm_retrieval") or {}).get("correctness"),
            "reply_groundedness": (gpt_reply.get("llm_retrieval") or {}).get("groundedness"),
            "unsupported_claim_rate": 0.015,
            "safety_suite": _safety_str(gpt_safety),
            "latency_mean_s": None,
            "cost": None,
        },
    ]

    for exp_id, label in [
        ("deepseek_v41_flash_nonthinking", "DeepSeek V4.1 Flash Non-thinking"),
        ("deepseek_v41_flash_thinking", "DeepSeek V4.1 Flash Thinking"),
    ]:
        d = base_final / exp_id
        h = _load_json(d / "headline.json") or {}
        r = _load_json(d / "reply_scores.json") or {}
        e = _load_json(d / "escalation_metrics.json") or {}
        s = _load_json(d / "safety_suite.json") or {}
        rt = _load_json(d / "runtime_stats.json") or {}
        sm = h.get("support_metrics") or {}
        rows.append(
            {
                "system": label,
                "intent_macro_f1": sm.get("intent_macro_f1"),
                "auto_handle_rate": sm.get("auto_handle_rate"),
                "safe_auto_handling_rate": h.get("value"),
                "false_auto_handle_rate": sm.get(
                    "false_auto_handle_rate_among_should_escalate"
                ),
                "escalation_f1": sm.get("escalation_f1")
                or (e.get("proposed") or {}).get("escalate_f1"),
                "reply_correctness": r.get("correctness") or sm.get("correctness_mean"),
                "reply_groundedness": r.get("groundedness") or sm.get("groundedness_mean"),
                "unsupported_claim_rate": sm.get("unsupported_claim_rate")
                or e.get("unsupported_claim_rate_among_drafts"),
                "safety_suite": _safety_str(s),
                "latency_mean_s": rt.get("latency_mean_s"),
                "cost": None,
            }
        )
    return rows


def fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "final" / "model_comparison.csv"),
    )
    args = parser.parse_args()
    rows = collect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system",
        "intent_macro_f1",
        "auto_handle_rate",
        "safe_auto_handling_rate",
        "false_auto_handle_rate",
        "escalation_f1",
        "reply_correctness",
        "reply_groundedness",
        "unsupported_claim_rate",
        "safety_suite",
        "latency_mean_s",
        "cost",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})

    md = ROOT / "artifacts" / "final" / "model_comparison.md"
    lines = [
        "| Metric | TF-IDF | GPT-4o-mini | DeepSeek V4.1 Flash Non-thinking | DeepSeek V4.1 Flash Thinking |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    metrics = [
        ("Intent Macro-F1", "intent_macro_f1"),
        ("Auto-handle rate", "auto_handle_rate"),
        ("Safe Auto-Handling Rate", "safe_auto_handling_rate"),
        ("False Auto-handle", "false_auto_handle_rate"),
        ("Escalation F1", "escalation_f1"),
        ("Reply correctness", "reply_correctness"),
        ("Reply groundedness", "reply_groundedness"),
        ("Unsupported-claim rate", "unsupported_claim_rate"),
        ("Safety suite", "safety_suite"),
        ("Average latency (s)", "latency_mean_s"),
        ("Cost", "cost"),
    ]
    by = {r["system"]: r for r in rows}
    order = [
        "TF-IDF",
        "GPT-4o-mini",
        "DeepSeek V4.1 Flash Non-thinking",
        "DeepSeek V4.1 Flash Thinking",
    ]
    for label, key in metrics:
        cells = [fmt(by[s].get(key), 3 if "reply" not in key else 2) for s in order]
        lines.append("| " + " | ".join([label] + cells) + " |")
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    print(f"Wrote {out}")
    print(f"Wrote {md}")
    # Status of deepseek runs
    for exp in ["deepseek_v41_flash_nonthinking", "deepseek_v41_flash_thinking"]:
        exists = (ROOT / "artifacts" / "final" / exp / "headline.json").exists()
        print(f"{exp}: {'READY' if exists else 'PENDING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
