#!/usr/bin/env python3
"""Build historical_cases.parquet from development messages (golden excluded)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.cases import (  # noqa: E402
    build_historical_cases,
    exclude_golden_from_messages,
)
from resolveflow.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    dev_path = resolve_path(cfg, cfg["data"]["development_messages"])
    golden_path = resolve_path(cfg, cfg["data"]["golden_path"])
    out_path = resolve_path(cfg, cfg["data"]["historical_cases"])
    labeled_path = resolve_path(cfg, cfg["data"]["labeled_dev"])

    messages = pd.read_parquet(dev_path)
    golden = pd.read_csv(golden_path)
    messages = exclude_golden_from_messages(messages, golden)
    print(f"Development messages after golden exclusion: {len(messages):,}")

    max_cases = args.max_cases or int(cfg["retrieval"]["max_cases"])
    cases = build_historical_cases(
        messages,
        brand=cfg["brand"]["name"],
        max_cases=max_cases,
        seed=int(cfg["evaluation"]["seed"]),
        prefer_english=True,
        label_intents=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(cases):,} cases)")

    # Larger silver-labeled set for TF-IDF training (still development-only)
    train_pool = build_historical_cases(
        messages,
        brand=cfg["brand"]["name"],
        max_cases=min(50000, len(messages)),
        seed=int(cfg["evaluation"]["seed"]) + 1,
        prefer_english=True,
        label_intents=True,
    )
    # Keep rows with non-empty intent
    train_pool = train_pool[train_pool["intent"].astype(str).str.len() > 0]
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    train_pool.to_parquet(labeled_path, index=False)
    print(f"Wrote labeled development pool: {labeled_path} ({len(train_pool):,})")
    print(train_pool["intent"].value_counts().head(12).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
