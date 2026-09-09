#!/usr/bin/env python3
"""Build (or load cached) retrieval embeddings + index from historical cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.config import load_config, resolve_path  # noqa: E402
from resolveflow.golden import load_golden_set  # noqa: E402
from resolveflow.retrieval.index import build_index_from_cases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rcfg = cfg["retrieval"]

    cases_path = resolve_path(cfg, cfg["data"]["historical_cases"])
    golden = load_golden_set(resolve_path(cfg, cfg["data"]["golden_path"]))
    cases = pd.read_parquet(cases_path)

    # Hard exclusion of any golden conversation IDs
    g_convs = set(golden["conversation_id"].astype(str))
    before = len(cases)
    cases = cases[~cases["conversation_id"].astype(str).isin(g_convs)].reset_index(
        drop=True
    )
    print(f"Cases: {before:,} → {len(cases):,} after golden conversation exclusion")

    emb_path = resolve_path(cfg, rcfg["embeddings_path"])
    meta_path = resolve_path(cfg, rcfg["metadata_path"])
    prev_fp = None
    if meta_path.exists():
        prev_fp = json.loads(meta_path.read_text()).get("fingerprint")

    index = build_index_from_cases(
        cases,
        embedding_model=rcfg["embedding_model"],
        brand=cfg["brand"]["name"],
        preprocessing_version=rcfg["preprocessing_version"],
        force_recompute=args.force,
        cache_embeddings_path=emb_path if emb_path.exists() else None,
        cache_fingerprint=prev_fp,
    )

    index.save(
        embeddings_path=emb_path,
        index_path=resolve_path(cfg, rcfg["index_path"]),
        metadata_path=meta_path,
        case_ids_path=resolve_path(cfg, rcfg["cases_id_map_path"]),
    )
    # Persist filtered cases used for index (aligned)
    cases.to_parquet(cases_path, index=False)
    print(f"Index built: {index.metadata['num_cases']} cases")
    print(f"Model: {index.metadata['embedding_model']}")
    print(f"Dim: {index.metadata['embedding_dimension']}")
    print(f"Fingerprint: {index.metadata['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
