#!/usr/bin/env python3
"""CLI demo for historical-case retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.config import load_config, resolve_path  # noqa: E402
from resolveflow.retrieval.evidence import format_evidence  # noqa: E402
from resolveflow.retrieval.retrieve import load_retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--evidence", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rcfg = cfg["retrieval"]
    retriever = load_retriever(
        embeddings_path=resolve_path(cfg, rcfg["embeddings_path"]),
        metadata_path=resolve_path(cfg, rcfg["metadata_path"]),
        case_ids_path=resolve_path(cfg, rcfg["cases_id_map_path"]),
        cases_path=resolve_path(cfg, cfg["data"]["historical_cases"]),
    )
    top_k = args.top_k or int(rcfg["top_k"])
    thr = args.threshold if args.threshold is not None else float(rcfg["similarity_threshold"])
    hits = retriever.retrieve(
        args.text, context=args.context, top_k=top_k, similarity_threshold=thr
    )

    print("Query")
    print("-----")
    print(args.text)
    if args.context:
        print("\nContext")
        print("-------")
        print(args.context)
    print()
    print("Results")
    print("-------")
    if not hits:
        print("No sufficiently similar historical case found.")
        return 0
    for i, h in enumerate(hits, start=1):
        print(f"\n{i}. similarity={h.similarity:.3f}")
        print(f"   intent={h.intent}")
        print(f'   customer="{h.customer_message[:200]}"')
        print(f'   response="{h.brand_response[:200]}"')
    if args.evidence:
        print("\nEvidence block")
        print("--------------")
        print(format_evidence(hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
