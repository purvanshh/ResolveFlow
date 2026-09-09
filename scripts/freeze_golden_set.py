#!/usr/bin/env python3
"""Freeze the golden set: validate, checksum, write manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.golden import freeze_golden_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = freeze_golden_set(
            golden_path=ROOT / "data" / "golden" / "golden_set.csv",
            manifest_path=ROOT / "data" / "golden" / "golden_set_manifest.json",
            sampling_method=(
                "stratified keyword pools + hard/short/long-thread oversample + "
                "rare-intent top-up; English-preferring filter; solo annotation "
                "per ANNOTATION_GUIDE with rule-assisted draft + manual overrides"
            ),
            random_seed=args.seed,
            dataset_version="twcs.csv+AmazonHelp",
            extra={"brand": "AmazonHelp", "annotator": "solo_author"},
        )
    except ValueError as exc:
        print(exc)
        return 1
    print("Golden set frozen")
    print(f"  examples: {manifest['num_examples']}")
    print(f"  sha256:   {manifest['sha256']}")
    print(f"  status:   {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
