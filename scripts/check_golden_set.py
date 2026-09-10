#!/usr/bin/env python3
"""Verify frozen golden set integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.golden import (  # noqa: E402
    load_golden_set,
    sha256_file,
    validate_golden_labels,
    verify_frozen_checksum,
)
from resolveflow.taxonomy import load_taxonomy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden", default=str(ROOT / "data" / "golden" / "golden_set.csv")
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data" / "golden" / "golden_set_manifest.json"),
    )
    args = parser.parse_args()
    path = Path(args.golden)
    manifest_path = Path(args.manifest)
    df = load_golden_set(path)
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    errors = validate_golden_labels(df, tax, require_complete=True)
    checksum = sha256_file(path) if path.exists() else ""
    frozen = verify_frozen_checksum(path, manifest_path)
    status = "FROZEN" if frozen and not errors else ("FAIL" if errors else "UNFROZEN")

    print("Golden Set")
    print("----------")
    print()
    print(f"Examples: {len(df)}")
    print(f"Checksum: {checksum}")
    print()
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        print(f"Manifest status: {man.get('status')}")
        print(f"Manifest sha256: {man.get('sha256')}")
    print()
    if errors:
        print("Errors:")
        for e in errors[:20]:
            print(f"  - {e}")
    print(f"Status: {status}")
    return 0 if status == "FROZEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
