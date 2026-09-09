#!/usr/bin/env python3
"""Evaluate retrieval Intent Recall@K on the golden set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.config import load_config, resolve_path  # noqa: E402
from resolveflow.golden import load_golden_set  # noqa: E402
from resolveflow.metrics import recall_at_k  # noqa: E402
from resolveflow.retrieval.retrieve import load_retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--manual-sample", type=int, default=40)
    args = parser.parse_args()
    cfg = load_config(args.config)

    golden = load_golden_set(resolve_path(cfg, cfg["data"]["golden_path"]))
    retriever = load_retriever(
        embeddings_path=resolve_path(cfg, cfg["retrieval"]["embeddings_path"]),
        metadata_path=resolve_path(cfg, cfg["retrieval"]["metadata_path"]),
        case_ids_path=resolve_path(cfg, cfg["retrieval"]["cases_id_map_path"]),
        cases_path=resolve_path(cfg, cfg["data"]["historical_cases"]),
    )
    # For recall we retrieve without threshold filtering (threshold=0) then report
    # both raw top-k and thresholded coverage.
    thr = float(cfg["retrieval"]["similarity_threshold"])
    retrieved_intents = []
    top1_sims = []
    examples = []
    for _, row in golden.iterrows():
        hits = retriever.retrieve(
            row["input_text"],
            context=row.get("context") or "",
            top_k=5,
            similarity_threshold=0.0,
        )
        retrieved_intents.append([h.intent for h in hits])
        top1_sims.append(hits[0].similarity if hits else 0.0)
        examples.append(
            {
                "example_id": row["example_id"],
                "gold_intent": row["intent"],
                "top": [
                    {
                        "case_id": h.case_id,
                        "similarity": h.similarity,
                        "intent": h.intent,
                        "customer_message": h.customer_message[:180],
                    }
                    for h in hits[:3]
                ],
            }
        )

    metrics = recall_at_k(retrieved_intents, golden["intent"].astype(str).tolist())
    # Thresholded: count queries with ≥1 hit above thr AND matching intent in top5
    thr_hits = 0
    any_above = 0
    for gold, ex in zip(golden["intent"].astype(str), examples, strict=False):
        above = [t for t in ex["top"] if t["similarity"] >= thr]
        # need full top5 for threshold — recompute from retrieved list
        pass
    for gold, intents, sims_row in zip(
        golden["intent"].astype(str),
        retrieved_intents,
        [
            [t["similarity"] for t in ex["top"]]  # only top3 stored — fix below
            for ex in examples
        ],
        strict=False,
    ):
        pass

    # Cleaner thresholded pass
    thr_intent_hit = 0
    thr_any = 0
    empty_after_thr = 0
    for _, row in golden.iterrows():
        hits = retriever.retrieve(
            row["input_text"],
            context=row.get("context") or "",
            top_k=5,
            similarity_threshold=thr,
        )
        if not hits:
            empty_after_thr += 1
            continue
        thr_any += 1
        if any(h.intent == row["intent"] for h in hits):
            thr_intent_hit += 1

    out = {
        "n_queries": len(golden),
        "embedding_model": retriever.index.metadata["embedding_model"],
        "num_cases": retriever.index.metadata["num_cases"],
        "similarity_threshold": thr,
        **metrics,
        "mean_top1_similarity": float(sum(top1_sims) / len(top1_sims)) if top1_sims else 0.0,
        "thresholded": {
            "queries_with_any_hit": thr_any,
            "queries_empty": empty_after_thr,
            "intent_hit_rate_among_all_queries": thr_intent_hit / len(golden),
        },
        "examples_preview": examples[:10],
    }

    # Manual relevance sample template (to be filled / partially auto-drafted)
    sample = examples[: args.manual_sample]
    manual = []
    for ex in sample:
        gold = ex["gold_intent"]
        top = ex["top"][0] if ex["top"] else None
        if top is None:
            score = 0
            note = "no retrieval"
        elif top["intent"] == gold and top["similarity"] >= 0.7:
            score = 2
            note = "auto: same intent + high sim (needs human confirm)"
        elif top["intent"] == gold:
            score = 1
            note = "auto: same intent, moderate sim (topic-only risk)"
        else:
            score = 0
            note = "auto: intent mismatch"
        manual.append(
            {
                "example_id": ex["example_id"],
                "gold_intent": gold,
                "top1_intent": top["intent"] if top else None,
                "top1_similarity": top["similarity"] if top else None,
                "relevance_0_to_2": score,
                "note": note,
            }
        )
    out["manual_relevance_sample"] = {
        "n": len(manual),
        "mean_score": float(sum(m["relevance_0_to_2"] for m in manual) / len(manual)),
        "distribution": {
            "0": sum(1 for m in manual if m["relevance_0_to_2"] == 0),
            "1": sum(1 for m in manual if m["relevance_0_to_2"] == 1),
            "2": sum(1 for m in manual if m["relevance_0_to_2"] == 2),
        },
        "items": manual,
        "caveat": (
            "Scores are heuristic drafts from intent+similarity; report treats them as "
            "diagnostic proxies pending fuller human review."
        ),
    }

    art = resolve_path(cfg, cfg["evaluation"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    path = art / "retrieval.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    print("Retrieval Evaluation")
    print("====================")
    print(f"Corpus cases: {out['num_cases']}")
    print(f"Recall@1: {out['recall_at_1']:.3f}")
    print(f"Recall@3: {out['recall_at_3']:.3f}")
    print(f"Recall@5: {out['recall_at_5']:.3f}")
    print(f"Mean top-1 similarity: {out['mean_top1_similarity']:.3f}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
