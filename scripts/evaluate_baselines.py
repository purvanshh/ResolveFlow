#!/usr/bin/env python3
"""Evaluate Phase 3 baselines on the frozen golden set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.baselines import (  # noqa: E402
    AlwaysEscalateBaseline,
    MajorityBaseline,
    NearestNeighborBaseline,
    TfidfBaseline,
    majority_label,
)
from resolveflow.config import load_config, resolve_path  # noqa: E402
from resolveflow.golden import load_golden_set  # noqa: E402
from resolveflow.metrics import escalation_metrics, intent_metrics  # noqa: E402
from resolveflow.retrieval.retrieve import load_retriever  # noqa: E402
from resolveflow.taxonomy import intent_names, load_taxonomy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    trivial = cfg["baseline"]["trivial_reply"]
    tax = load_taxonomy(ROOT / "configs" / "intents.yaml")
    labels = intent_names(tax)

    golden = load_golden_set(resolve_path(cfg, cfg["data"]["golden_path"]))
    # Majority class from silver training labels (development), not golden
    labeled = pd.read_parquet(resolve_path(cfg, cfg["data"]["labeled_dev"]))
    maj = majority_label(labeled["intent"])

    train_meta = json.loads(
        (
            resolve_path(cfg, cfg["evaluation"]["artifacts_dir"]) / "baseline_train.json"
        ).read_text()
    )
    thr = float(train_meta["chosen_confidence_threshold"])

    tfidf = TfidfBaseline.load(resolve_path(cfg, cfg["baseline"]["model_path"]))
    tfidf.confidence_threshold = thr
    tfidf.trivial_reply = trivial

    retriever = load_retriever(
        embeddings_path=resolve_path(cfg, cfg["retrieval"]["embeddings_path"]),
        metadata_path=resolve_path(cfg, cfg["retrieval"]["metadata_path"]),
        case_ids_path=resolve_path(cfg, cfg["retrieval"]["cases_id_map_path"]),
        cases_path=resolve_path(cfg, cfg["data"]["historical_cases"]),
    )
    sim_thr = float(cfg["retrieval"]["similarity_threshold"])

    def retrieve_fn(query, top_k=1, similarity_threshold=sim_thr):
        # query may already include context from combine_text
        return retriever.retrieve(
            query, context="", top_k=top_k, similarity_threshold=similarity_threshold
        )

    systems = {
        "majority": MajorityBaseline(maj, trivial),
        "always_escalate": AlwaysEscalateBaseline(maj, trivial),
        "tfidf_logistic_regression": tfidf,
        "nearest_neighbor_response": NearestNeighborBaseline(
            retrieve_fn,
            trivial_reply=trivial,
            similarity_threshold=sim_thr,
        ),
    }

    y_true_intent = golden["intent"].astype(str).tolist()
    y_true_esc = golden["escalation_expected"].astype(str).tolist()

    results: dict = {}
    for name, system in systems.items():
        pred_intents = []
        pred_esc = []
        for _, row in golden.iterrows():
            p = system.predict(row["input_text"], row.get("context") or "")
            pred_intents.append(p.intent)
            pred_esc.append(p.escalation)
        intent_m = intent_metrics(y_true_intent, pred_intents, labels=labels)
        esc_m = escalation_metrics(y_true_esc, pred_esc)
        results[name] = {
            "intent": {
                "accuracy": intent_m["accuracy"],
                "macro_f1": intent_m["macro_f1"],
                "weighted_f1": intent_m["weighted_f1"],
                "per_intent": intent_m["per_intent"],
                "confusion_matrix": intent_m["confusion_matrix"],
            },
            "escalation": esc_m,
            "notes": {
                "majority_intent": maj if name == "majority" else None,
                "confidence_threshold": thr if name == "tfidf_logistic_regression" else None,
            },
        }

    # Confidence threshold curve on golden using TF-IDF confidences
    curves = []
    confs = []
    tfidf_intents = []
    for _, row in golden.iterrows():
        intent, conf = tfidf.predict_intent(row["input_text"], row.get("context") or "")
        tfidf_intents.append(intent)
        confs.append(conf)
    for thr_i in cfg["baseline"]["confidence_thresholds"]:
        pred_esc = [
            "auto_handle" if c >= float(thr_i) else "escalate" for c in confs
        ]
        esc_m = escalation_metrics(y_true_esc, pred_esc)
        curves.append({"threshold": float(thr_i), **esc_m})

    out = {
        "brand": cfg["brand"]["name"],
        "n_golden": len(golden),
        "systems": results,
        "confidence_threshold_curves": curves,
        "chosen_confidence_threshold": thr,
        "train_val_summary": {
            "val_accuracy": train_meta.get("val_accuracy"),
            "val_macro_f1": train_meta.get("val_macro_f1"),
        },
    }

    art = resolve_path(cfg, cfg["evaluation"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    path = art / "baselines.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    print("Intent Classification")
    print("=====================")
    print()
    print(f"{'System':28s} {'Accuracy':>10s} {'Macro F1':>10s}")
    for name in [
        "majority",
        "tfidf_logistic_regression",
        "nearest_neighbor_response",
    ]:
        m = results[name]["intent"]
        print(f"{name:28s} {m['accuracy']:10.3f} {m['macro_f1']:10.3f}")
    print()
    print("Escalation (always_escalate)")
    ae = results["always_escalate"]["escalation"]
    print(f"  Auto-handle rate: {ae['auto_handle_rate']:.3f}")
    print(f"  False-auto-handle rate: {ae['false_auto_handle_rate']:.3f}")
    print(f"  Escalate F1: {ae['escalate_f1']:.3f}")
    print()
    print("Escalation (TF-IDF confidence threshold)")
    te = results["tfidf_logistic_regression"]["escalation"]
    print(f"  Threshold: {thr}")
    print(f"  Auto-handle rate: {te['auto_handle_rate']:.3f}")
    print(f"  False-auto-handle rate: {te['false_auto_handle_rate']:.3f}")
    print(f"  Escalate F1: {te['escalate_f1']:.3f}")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
