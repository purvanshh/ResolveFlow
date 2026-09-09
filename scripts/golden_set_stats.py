#!/usr/bin/env python3
"""Write golden-set statistics artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.golden import load_golden_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden", default=str(ROOT / "data" / "golden" / "golden_set.csv")
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "golden_set_stats.json"),
    )
    args = parser.parse_args()

    df = load_golden_set(args.golden)
    lengths = df["input_text"].astype(str).str.len()
    ctx_lens = df["context"].astype(str).str.len()
    stats = {
        "num_examples": int(len(df)),
        "intent_distribution": dict(Counter(df["intent"].astype(str))),
        "escalation_distribution": dict(
            Counter(df["escalation_expected"].astype(str))
        ),
        "difficulty_distribution": dict(Counter(df["difficulty"].astype(str))),
        "message_length": {
            "mean": float(lengths.mean()) if len(df) else 0.0,
            "median": float(lengths.median()) if len(df) else 0.0,
            "min": int(lengths.min()) if len(df) else 0,
            "max": int(lengths.max()) if len(df) else 0,
        },
        "context_length": {
            "mean": float(ctx_lens.mean()) if len(df) else 0.0,
            "median": float(ctx_lens.median()) if len(df) else 0.0,
            "empty_fraction": float((ctx_lens == 0).mean()) if len(df) else 0.0,
        },
        "unique_conversations": int(df["conversation_id"].nunique()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")

    print("Golden Evaluation Set")
    print("=====================")
    print()
    print(f"Examples: {stats['num_examples']}")
    print()
    print("Intent distribution:")
    for k, v in sorted(stats["intent_distribution"].items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print()
    print("Escalation:")
    esc = stats["escalation_distribution"]
    print(f"  Auto-handle: {esc.get('auto_handle', 0)}")
    print(f"  Escalate: {esc.get('escalate', 0)}")
    print(f"  Uncertain: {esc.get('uncertain', 0)}")
    print()
    print("Difficulty:")
    diff = stats["difficulty_distribution"]
    print(f"  Easy: {diff.get('easy', 0)}")
    print(f"  Medium: {diff.get('medium', 0)}")
    print(f"  Hard: {diff.get('hard', 0)}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
