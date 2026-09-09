#!/usr/bin/env python3
"""Profile the Customer Support on Twitter dataset and select a brand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.data.ingest import load_raw_tweets  # noqa: E402
from resolveflow.data.profile import (  # noqa: E402
    brand_comparison_markdown,
    compute_brand_table,
    format_profile_report,
    profile_dataframe,
    select_brand,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile ResolveFlow raw data")
    parser.add_argument(
        "--path",
        default=str(ROOT / "data" / "raw" / "twcs.csv"),
        help="Path to twcs.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of brands to compare by reply volume",
    )
    parser.add_argument(
        "--write-interim",
        action="store_true",
        help="Write brand metrics JSON under data/interim/",
    )
    args = parser.parse_args()

    print("Loading dataset...", flush=True)
    df = load_raw_tweets(args.path)
    print(f"Loaded {len(df):,} rows", flush=True)

    print("Profiling schema...", flush=True)
    profile = profile_dataframe(df)

    print("Computing brand conversation metrics (this may take a few minutes)...", flush=True)
    metrics = compute_brand_table(df, top_n=args.top_n)
    selected = select_brand(metrics)

    report = format_profile_report(profile, metrics, selected, top_k=min(10, len(metrics)))
    print()
    print(report)

    # Also emit the classic "Dataset profile" block for Step 3
    print("Dataset profile")
    print("---------------")
    print()
    print(f"Rows: {profile['rows']:,}")
    print(f"Columns: {profile['columns']}")
    print()
    print("Column names:")
    for c in profile["column_names"]:
        print(f"  {c}")
    print()
    print("Missing values:")
    for c, n in profile["missing_values"].items():
        print(f"  {c}: {n:,}")
    print()
    print(f"Unique brands: {profile['unique_brands']}")
    print()
    print(
        f"Date range: {profile['date_range']['min']} → {profile['date_range']['max']}"
    )
    print()
    print("Inbound distribution:")
    for k, v in profile["inbound_distribution"].items():
        print(f"  {k}: {v:,}")
    print()
    print("Brand comparison (markdown)")
    print("---------------------------")
    print(brand_comparison_markdown(metrics, limit=20))

    if args.write_interim and selected is not None:
        interim = ROOT / "data" / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        payload = {
            "selected_brand": selected.brand,
            "profile": {
                k: v
                for k, v in profile.items()
                if k != "brands"  # long list
            },
            "brand_count": profile["unique_brands"],
            "metrics": [m.as_dict() for m in metrics],
        }
        out = interim / "phase1_brand_metrics.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out}")

        # Persist selection for later phases
        config_path = ROOT / "configs" / "brand.yaml"
        config_path.write_text(
            f'selected_brand: "{selected.brand}"\n'
            f"usable_conversations: {selected.usable_conversations}\n"
            f"response_available: {selected.response_available}\n"
        )
        print(f"Wrote {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
