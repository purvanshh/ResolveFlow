#!/usr/bin/env python3
"""Validate intents.yaml against the golden set."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.golden import load_golden_set, validate_golden_labels  # noqa: E402
from resolveflow.taxonomy import load_taxonomy, validate_taxonomy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default=str(ROOT / "configs" / "intents.yaml"))
    parser.add_argument("--golden", default=str(ROOT / "data" / "golden" / "golden_set.csv"))
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    tax_errors = validate_taxonomy(taxonomy)
    golden = load_golden_set(args.golden)
    gold_errors = validate_golden_labels(golden, taxonomy, require_complete=True)

    print("Taxonomy validation")
    print("===================")
    print()
    print(f"Intents: {len(taxonomy.get('intents', []))}")
    print(f"Golden examples: {len(golden)}")
    print()
    print("Examples per intent:")
    counts = Counter(golden["intent"].astype(str))
    names = [i["name"] for i in taxonomy["intents"]]
    warnings: list[str] = []
    for name in names:
        n = counts.get(name, 0)
        print(f"  {name:36s} {n:4d}")
        if n < 5:
            warnings.append(f"{name} has only {n} examples")
    extras = set(counts) - set(names) - {""}
    for e in sorted(extras):
        warnings.append(f"golden contains unknown intent '{e}'")
        print(f"  {e:36s} {counts[e]:4d}  (UNKNOWN)")

    print()
    if tax_errors:
        print("Taxonomy errors:")
        for e in tax_errors:
            print(f"  - {e}")
    if gold_errors:
        print("Golden errors:")
        for e in gold_errors[:30]:
            print(f"  - {e}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    fatal = bool(tax_errors or gold_errors)
    if fatal:
        print("\nStatus: FAIL")
        return 1
    if warnings:
        print("\nStatus: PASS WITH WARNINGS")
        return 0
    print("\nStatus: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
