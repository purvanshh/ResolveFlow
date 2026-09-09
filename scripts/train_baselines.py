#!/usr/bin/env python3
"""Train TF-IDF + LR baseline on silver-labeled development data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.baselines import (  # noqa: E402
    TfidfBaseline,
    combine_text,
    majority_label,
)
from resolveflow.config import load_config, resolve_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["evaluation"]["seed"])

    labeled = pd.read_parquet(resolve_path(cfg, cfg["data"]["labeled_dev"]))
    # Downsample dominant other_unclear so the classifier isn't a trivial majority model
    other = labeled[labeled["intent"] == "other_unclear"]
    rest = labeled[labeled["intent"] != "other_unclear"]
    cap = min(len(other), max(len(rest), 1))
    # Cap other_unclear at median class size among non-other (or rest size)
    non_other_counts = rest["intent"].value_counts()
    target_other = int(min(len(other), max(int(non_other_counts.median()), 2000)))
    other = other.sample(n=min(len(other), target_other), random_state=seed)
    # Upsample rare classes lightly by sampling with replacement up to p25 of rest
    parts = [other]
    min_n = max(200, int(non_other_counts.quantile(0.25))) if len(non_other_counts) else 200
    for intent, g in rest.groupby("intent"):
        if len(g) < min_n:
            parts.append(g.sample(n=min_n, replace=True, random_state=seed))
        else:
            parts.append(g)
    labeled = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)

    texts = [
        combine_text(r.customer_message, r.customer_context)
        for r in labeled.itertuples(index=False)
    ]
    labels = labeled["intent"].astype(str).tolist()

    split = float(cfg["baseline"]["train_val_split"])
    # Stratify only if every class has >=2 samples
    vc = pd.Series(labels).value_counts()
    stratify = labels if vc.min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        train_size=split,
        random_state=seed,
        stratify=stratify,
    )

    tfidf_cfg = cfg["baseline"]["tfidf"]
    ngram = tuple(tfidf_cfg["ngram_range"])
    model = TfidfBaseline.train(
        X_train,
        y_train,
        max_features=int(tfidf_cfg["max_features"]),
        ngram_range=(int(ngram[0]), int(ngram[1])),
        min_df=int(tfidf_cfg["min_df"]),
        C=float(tfidf_cfg["C"]),
        max_iter=int(tfidf_cfg["max_iter"]),
        trivial_reply=cfg["baseline"]["trivial_reply"],
        confidence_threshold=0.7,
    )

    # Tune confidence threshold on validation for escalation tradeoff
    from sklearn.metrics import f1_score

    val_pred = []
    val_conf = []
    for t in X_val:
        intent, conf = model.predict_intent(t)
        # predict_intent with combined text already — pass as message only
        val_pred.append(intent)
        val_conf.append(conf)
    # Recompute properly
    val_pred, val_conf = [], []
    for t in X_val:
        # texts already combined
        proba = model.pipeline.predict_proba([t])[0]
        idx = int(proba.argmax())
        val_pred.append(str(model.pipeline.classes_[idx]))
        val_conf.append(float(proba[idx]))

    macro = f1_score(y_val, val_pred, average="macro")
    acc = float((pd.Series(val_pred) == pd.Series(y_val)).mean())

    thresholds = list(cfg["baseline"]["confidence_thresholds"])
    # Need escalation labels on val — approximate: escalate if silver intent is
    # in high-risk set OR use gold-like heuristic from annotate defaults.
    # For threshold tuning we optimize: prefer high conf auto-handle when
    # prediction matches silver label (proxy for "safe automation").
    curves = []
    for thr in thresholds:
        auto = [c >= thr for c in val_conf]
        # false auto-handle: auto AND predicted intent != silver label
        false_auto = sum(
            1 for a, p, y in zip(auto, val_pred, y_val, strict=False) if a and p != y
        )
        n_auto = sum(auto)
        curves.append(
            {
                "threshold": thr,
                "auto_handle_rate": n_auto / len(val_conf) if val_conf else 0.0,
                "false_auto_handle_rate": (false_auto / n_auto) if n_auto else 0.0,
                "n_auto": n_auto,
            }
        )

    # Choose threshold with false_auto_handle_rate <= 0.25 and max auto rate;
    # fallback to 0.7
    viable = [c for c in curves if c["false_auto_handle_rate"] <= 0.25]
    if viable:
        best = max(viable, key=lambda c: c["auto_handle_rate"])
    else:
        best = min(curves, key=lambda c: c["false_auto_handle_rate"])
    model.confidence_threshold = float(best["threshold"])

    model_path = resolve_path(cfg, cfg["baseline"]["model_path"])
    model.save(model_path)

    meta = {
        "majority_intent": majority_label(labels),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "val_accuracy": acc,
        "val_macro_f1": float(macro),
        "chosen_confidence_threshold": model.confidence_threshold,
        "threshold_curves": curves,
        "model_path": str(model_path),
    }
    out = resolve_path(cfg, cfg["evaluation"]["artifacts_dir"]) / "baseline_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"Saved model → {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
