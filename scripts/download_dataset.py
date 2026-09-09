#!/usr/bin/env python3
"""Download the Customer Support on Twitter dataset into data/raw/."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "twcs.csv"

# Public mirror of the Kaggle thoughtvector/customer-support-on-twitter CSV.
HF_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--url", default=HF_URL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.force:
        print(f"Already present: {args.out} ({args.out.stat().st_size:,} bytes)")
        print("Pass --force to re-download.")
        return 0

    print(f"Downloading {args.url}")
    print(f"→ {args.out}")
    urllib.request.urlretrieve(args.url, args.out)
    print(f"Done ({args.out.stat().st_size:,} bytes)")
    print("Prefer Kaggle when credentials are available:")
    print(
        "  kaggle datasets download -d thoughtvector/customer-support-on-twitter "
        "-p data/raw --unzip"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
