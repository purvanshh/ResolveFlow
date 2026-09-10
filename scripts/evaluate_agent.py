#!/usr/bin/env python3
"""Evaluate the agent on the frozen golden set + ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.agent.factory import build_agent  # noqa: E402
from resolveflow.agent.safety import validate_reply  # noqa: E402
from resolveflow.baselines import AlwaysEscalateBaseline, MajorityBaseline, majority_label  # noqa: E402
from resolveflow.config import load_config, resolve_path  # noqa: E402
from resolveflow.golden import load_golden_set  # noqa: E402
from resolveflow.metrics import escalation_metrics, intent_metrics  # noqa: E402
from resolveflow.schemas import AgentRequest  # noqa: E402
from resolveflow.taxonomy import intent_names, load_taxonomy  # noqa: E402


def _run_corpus(agent, golden, *, label: str = "agent"):
    rows = []
    n = len(golden)
    for i, (_, row) in enumerate(golden.iterrows()):
        if i % 10 == 0:
            print(f"[{label}] {i}/{n}", flush=True)
        d = agent.handle(
            AgentRequest(message=row["input_text"], context=row.get("context") or "")
        )
        rows.append(
            {
                "example_id": row["example_id"],
                "gold_intent": row["intent"],
                "predicted_intent": d.intent,
                "intent_correct": d.intent == row["intent"],
                "gold_escalation": row["escalation_expected"],
                "predicted_escalation": "escalate" if d.escalate else "auto_handle",
                "escalation_correct": (
                    (row["escalation_expected"] == "escalate" and d.escalate)
                    or (row["escalation_expected"] == "auto_handle" and not d.escalate)
                ),
                "confidence": d.confidence,
                "reply": d.reply,
                "evidence": [e.to_dict() for e in d.evidence],
                "safety_flags": d.safety_flags,
                "escalation_reason": d.escalation_reason,
                "customer_message": row["input_text"],
                "context": row.get("context") or "",
                "difficulty": row.get("difficulty") or "",
                "meta": d.meta,
            }
        )
    return rows


def _unsupported_rate(rows) -> float:
    n = 0
    for r in rows:
        reply = r.get("reply") or ""
        if not reply:
            continue
        safety = validate_reply(reply, customer_message=r.get("gold_intent", ""))
        # count monetary/timeline/action/policy flags on produced replies
        bad = {
            "UNSUPPORTED_MONETARY_CLAIM",
            "UNSUPPORTED_TIMELINE",
            "UNSUPPORTED_ACTION",
            "UNSUPPORTED_POLICY",
            "ACCOUNT_SPECIFIC_CLAIM",
        }
        if set(r.get("safety_flags") or []) & bad or set(safety.flags) & bad:
            n += 1
    with_reply = sum(1 for r in rows if r.get("reply"))
    return n / with_reply if with_reply else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--mode", default="offline", choices=["offline", "mock", "auto", "openai"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tax = load_taxonomy()
    labels = intent_names(tax)
    golden = load_golden_set(resolve_path(cfg, cfg["data"]["golden_path"]))
    if args.limit:
        golden = golden.head(args.limit)

    art = Path(resolve_path(cfg, "artifacts/evaluation"))
    metrics_dir = Path(resolve_path(cfg, cfg["evaluation"]["artifacts_dir"]))
    art.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = Path(resolve_path(cfg, "artifacts/examples"))
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Main agent
    agent, meta = build_agent(config_path=args.config, provider_mode=args.mode, top_k=3)
    rows = _run_corpus(agent, golden, label="main_k3")
    pred_path = art / "agent_predictions.jsonl"
    with pred_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    y_true = [r["gold_intent"] for r in rows]
    y_pred = [r["predicted_intent"] for r in rows]
    intent_m = intent_metrics(y_true, y_pred, labels=labels)
    esc_true = [r["gold_escalation"] for r in rows]
    esc_pred = [r["predicted_escalation"] for r in rows]
    # Map uncertain gold to escalate for binary metrics if present
    esc_true = ["escalate" if e == "uncertain" else e for e in esc_true]
    esc_m = escalation_metrics(esc_true, esc_pred)
    safety_rate = _unsupported_rate(rows)

    # Baselines for comparison table
    import pandas as pd

    labeled = pd.read_parquet(resolve_path(cfg, cfg["data"]["labeled_dev"]))
    maj = majority_label(labeled["intent"])
    trivial = cfg["baseline"]["trivial_reply"]
    majority = MajorityBaseline(maj, trivial)
    always = AlwaysEscalateBaseline(maj, trivial)
    base_intent = {}
    for name, system in [("majority", majority)]:
        preds = [system.predict_intent(r["input_text"], r.get("context") or "")[0] for _, r in golden.iterrows()]
        base_intent[name] = intent_metrics(y_true, preds, labels=labels)
    # load prior TF-IDF metrics if present
    prior = {}
    prior_path = metrics_dir / "baselines.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())

    always_esc = escalation_metrics(
        esc_true,
        ["escalate"] * len(rows),
    )

    agent_intent_path = metrics_dir / "agent_intent_metrics.json"
    agent_intent_path.write_text(
        json.dumps(
            {
                "runtime": meta,
                "n": len(rows),
                "intent": intent_m,
                "baseline_majority_macro_f1": base_intent["majority"]["macro_f1"],
                "baseline_tfidf_macro_f1": (
                    prior.get("systems", {})
                    .get("tfidf_logistic_regression", {})
                    .get("intent", {})
                    .get("macro_f1")
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (metrics_dir / "agent_escalation_metrics.json").write_text(
        json.dumps(
            {
                "proposed": esc_m,
                "always_escalate": always_esc,
                "chosen_note": "Proposed policy is risk-aware (high-risk intents, evidence, safety).",
            },
            indent=2,
        )
        + "\n"
    )
    (metrics_dir / "safety_metrics.json").write_text(
        json.dumps(
            {
                "unsupported_claim_rate_among_drafts": safety_rate,
                "n_with_draft": sum(1 for r in rows if r.get("reply")),
                "prompt_injection_suite": "see scripts/evaluate_safety.py",
            },
            indent=2,
        )
        + "\n"
    )

    # Ablations: K=0,1,3,5
    ablation = {}
    for k in [0, 1, 3, 5]:
        a, m = build_agent(
            config_path=args.config,
            provider_mode=args.mode,
            top_k=k,
            disable_retrieval=(k == 0),
        )
        arows = _run_corpus(a, golden, label=f"ablation_k{k}")
        if k == 0:
            with (art / "no_retrieval_predictions.jsonl").open("w") as f:
                for r in arows:
                    f.write(json.dumps(r) + "\n")
        ablation[f"k_{k}"] = {
            "intent_macro_f1": intent_metrics(
                [r["gold_intent"] for r in arows],
                [r["predicted_intent"] for r in arows],
                labels=labels,
            )["macro_f1"],
            "auto_handle_rate": escalation_metrics(
                ["escalate" if e == "uncertain" else e for e in [r["gold_escalation"] for r in arows]],
                [r["predicted_escalation"] for r in arows],
            )["auto_handle_rate"],
            "false_auto_handle_rate": escalation_metrics(
                ["escalate" if e == "uncertain" else e for e in [r["gold_escalation"] for r in arows]],
                [r["predicted_escalation"] for r in arows],
            )["false_auto_handle_rate"],
            "unsupported_claim_rate": _unsupported_rate(arows),
            "mean_evidence": float(
                sum(len(r["evidence"]) for r in arows) / len(arows)
            ),
            "draft_rate": sum(1 for r in arows if r.get("reply")) / len(arows),
        }
    (art / "retrieval_ablation.json").write_text(json.dumps(ablation, indent=2) + "\n")

    # Example packs
    good = [r for r in rows if r["intent_correct"] and r["escalation_correct"]][:5]
    bad = [r for r in rows if not r["intent_correct"] or not r["escalation_correct"]][:8]
    (examples_dir / "agent_good_examples.json").write_text(json.dumps(good, indent=2) + "\n")
    (examples_dir / "agent_failure_examples.json").write_text(json.dumps(bad, indent=2) + "\n")

    print("Agent evaluation")
    print("================")
    print(f"Runtime: {meta}")
    print(f"Intent Acc: {intent_m['accuracy']:.3f}")
    print(f"Intent Macro-F1: {intent_m['macro_f1']:.3f}")
    print(f"Intent Weighted-F1: {intent_m['weighted_f1']:.3f}")
    print(f"Majority Macro-F1: {base_intent['majority']['macro_f1']:.3f}")
    if prior:
        print(
            "TF-IDF Macro-F1:",
            prior["systems"]["tfidf_logistic_regression"]["intent"]["macro_f1"],
        )
    print(f"Proposed auto-handle rate: {esc_m['auto_handle_rate']:.3f}")
    print(f"Proposed false-auto-handle rate: {esc_m['false_auto_handle_rate']:.3f}")
    print(f"Proposed escalate F1: {esc_m['escalate_f1']:.3f}")
    print(f"Always-escalate F1: {always_esc['escalate_f1']:.3f}")
    print(f"Unsupported claim rate: {safety_rate:.3f}")
    print(f"Wrote {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
