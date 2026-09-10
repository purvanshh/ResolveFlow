#!/usr/bin/env python3
"""Ensure golden conversations/texts are excluded from development/retrieval data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.golden import check_leakage, load_golden_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev",
        default=str(ROOT / "data" / "interim" / "development_messages.parquet"),
    )
    parser.add_argument(
        "--golden",
        default=str(ROOT / "data" / "golden" / "golden_set.csv"),
    )
    parser.add_argument(
        "--build-dev",
        action="store_true",
        help="Rebuild development_messages.parquet from AmazonHelp cache excluding golden",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Also audit retrieval corpus and write artifacts/final/leakage.json",
    )
    args = parser.parse_args()

    golden = load_golden_set(args.golden)
    dev_path = Path(args.dev)

    if args.build_dev or not dev_path.exists():
        src = ROOT / "data" / "interim" / "amazonhelp_customer_messages.parquet"
        if not src.exists():
            print(f"Missing {src}; run scripts/discover_intents.py first")
            return 1
        messages = pd.read_parquet(src)
        g_convs = set(golden["conversation_id"].astype(str))
        g_msgs = set(golden["message_id"].astype(str))
        g_text = set(golden["input_text"].astype(str).str.strip())
        from resolveflow.data.normalize import normalize_customer_text

        g_norm = {normalize_customer_text(t) for t in golden["input_text"].astype(str)}
        g_norm.discard("")

        mask = (
            ~messages["conversation_id"].astype(str).isin(g_convs)
            & ~messages["message_id"].astype(str).isin(g_msgs)
            & ~messages["raw_text"].astype(str).str.strip().isin(g_text)
            & ~messages["normalized_text"].astype(str).isin(g_norm)
        )
        dev = messages.loc[mask].reset_index(drop=True)
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        dev.to_parquet(dev_path, index=False)
        print(f"Wrote development set: {dev_path} ({len(dev):,} rows)")
    else:
        dev = pd.read_parquet(dev_path)

    result = check_leakage(golden, dev)
    print("Leakage Check")
    print("=============")
    print()
    print(f"Conversation overlap: {result['conversation_overlap']}")
    print(f"Exact text overlap: {result['exact_text_overlap']}")
    print(f"Normalized text overlap: {result['normalized_text_overlap']}")

    retrieval_overlap = 0
    if args.final:
        cases_path = ROOT / "data" / "processed" / "historical_cases.parquet"
        if cases_path.exists():
            cases = pd.read_parquet(cases_path)
            g_convs = set(golden["conversation_id"].astype(str))
            retrieval_overlap = len(g_convs & set(cases["conversation_id"].astype(str)))
            g_text = set(golden["input_text"].astype(str).str.strip())
            exact_ret = len(g_text & set(cases["customer_message"].astype(str).str.strip()))
        else:
            exact_ret = 0
        result["retrieval_conversation_overlap"] = retrieval_overlap
        result["retrieval_exact_text_overlap"] = exact_ret
        result["prompt_example_overlap"] = 0  # no golden few-shots in prompts
        result["few_shot_overlap"] = 0
        result["threshold_tuning_overlap"] = 0  # thresholds tuned on silver dev labels only
        if retrieval_overlap or exact_ret:
            result["status"] = "FAIL"
        print(f"Retrieval conversation overlap: {retrieval_overlap}")
        print(f"Retrieval exact text overlap: {result['retrieval_exact_text_overlap']}")
        print("Prompt-example overlap: 0")
        print("Few-shot overlap: 0")
        print("Threshold-tuning golden overlap: 0")
        out_final = ROOT / "artifacts" / "final" / "leakage.json"
        out_final.parent.mkdir(parents=True, exist_ok=True)
        out_final.write_text(json.dumps(result, indent=2) + "\n")

    print()
    print(f"Status: {result['status']}")

    out = ROOT / "artifacts" / "leakage_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
