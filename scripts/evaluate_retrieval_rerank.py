#!/usr/bin/env python3
"""Offline ablation: semantic vs intent/resolution-aware reranking.

Uses frozen GPT predicted intents (inference-time) — never gold labels for ranking.
Recomputes Intent Recall@K and end-to-end escalation metrics under the calibrated
escalation policy without calling an LLM API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.agent.policy import EscalationPolicy  # noqa: E402
from resolveflow.agent.safety import detect_human_request  # noqa: E402
from resolveflow.config import load_config, resolve_path  # noqa: E402
from resolveflow.golden import load_golden_set  # noqa: E402
from resolveflow.metrics import escalation_metrics, recall_at_k  # noqa: E402
from resolveflow.retrieval.evidence import is_sufficient_evidence  # noqa: E402
from resolveflow.retrieval.rerank import rerank_cases  # noqa: E402
from resolveflow.retrieval.retrieve import load_retriever  # noqa: E402
from resolveflow.schemas import IntentPrediction, SafetyResult  # noqa: E402


HIGH_RISK = [
    "payment_billing",
    "refund_request",
    "refund_status",
    "account_access",
    "package_missing_or_misdelivered",
    "order_quality_issue",
]


def select_hits(retriever, message, context, pred_intent, mode: str, top_k: int, thr: float, pool: int):
    if mode == "none":
        # Mirror AgentPipeline baseline: intent filter then top_k + threshold.
        hits = retriever.retrieve(message, context=context, top_k=max(top_k * 4, 8), similarity_threshold=0.0)
        filtered = [h for h in hits if h.intent == pred_intent]
        if filtered:
            hits = [h for h in filtered if h.similarity >= thr][:top_k]
        else:
            hits = retriever.retrieve(
                message, context=context, top_k=top_k, similarity_threshold=thr
            )
        return hits

    hits = retriever.retrieve(message, context=context, top_k=pool, similarity_threshold=0.0)
    hits = rerank_cases(
        hits,
        query=f"{message}\n{context}".strip(),
        predicted_intent=pred_intent,
        mode=mode,
    )
    return [h for h in hits if h.similarity >= thr][:top_k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--pool", type=int, default=20)
    args = parser.parse_args()
    cfg = load_config(args.config)
    thr = float(cfg["retrieval"]["similarity_threshold"])
    top_k = 5  # for recall curve; agent uses 3 for e2e below

    golden = load_golden_set(resolve_path(cfg, cfg["data"]["golden_path"]))
    preds_path = ROOT / "artifacts" / "evaluation" / "agent_predictions.jsonl"
    pred_rows = {
        json.loads(l)["example_id"]: json.loads(l)
        for l in preds_path.read_text().splitlines()
        if l.strip()
    }

    retriever = load_retriever(
        embeddings_path=resolve_path(cfg, cfg["retrieval"]["embeddings_path"]),
        metadata_path=resolve_path(cfg, cfg["retrieval"]["metadata_path"]),
        case_ids_path=resolve_path(cfg, cfg["retrieval"]["cases_id_map_path"]),
        cases_path=resolve_path(cfg, cfg["data"]["historical_cases"]),
    )
    policy = EscalationPolicy(high_risk_intents=HIGH_RISK)

    modes = ["none", "intent", "resolution", "combined"]
    results = []

    for mode in modes:
        retrieved_intents = []
        e2e_preds = []
        flips = {"to_auto": 0, "to_esc": 0, "new_safe": 0, "new_fah": 0, "lost_safe": 0}
        fah_rows = []

        for _, row in golden.iterrows():
            eid = row["example_id"]
            pr = pred_rows[eid]
            message = pr.get("customer_message") or row["input_text"]
            context = pr.get("context") or row.get("context") or ""
            pred_intent = pr["predicted_intent"]
            conf = float(pr.get("confidence") or 0)
            flags = list(pr.get("safety_flags") or [])

            hits5 = select_hits(
                retriever, message, context, pred_intent, mode, top_k=5, thr=0.0, pool=args.pool
            )
            retrieved_intents.append([h.intent for h in hits5])

            # Agent-style evidence (k=3, threshold applied)
            hits3 = select_hits(
                retriever, message, context, pred_intent, mode, top_k=3, thr=thr, pool=args.pool
            )
            top_sim = float(hits3[0].similarity) if hits3 else 0.0
            sufficient = is_sufficient_evidence(hits3, similarity_threshold=thr)
            safety = SafetyResult(safe=len(flags) == 0, flags=flags)
            d = policy.decide(
                message=message,
                intent=IntentPrediction(intent=pred_intent, confidence=conf),
                top_similarity=top_sim,
                evidence_count=len(hits3),
                evidence_sufficient=sufficient,
                safety=safety,
            )
            new_esc = "escalate" if d.escalate else "auto_handle"
            e2e_preds.append(new_esc)

            old = pr["predicted_escalation"]
            if old != new_esc:
                if new_esc == "auto_handle":
                    flips["to_auto"] += 1
                    if (
                        pr["gold_escalation"] == "auto_handle"
                        and pr["intent_correct"]
                        and not flags
                    ):
                        flips["new_safe"] += 1
                    if pr["gold_escalation"] == "escalate":
                        flips["new_fah"] += 1
                else:
                    flips["to_esc"] += 1
                    if (
                        old == "auto_handle"
                        and pr["gold_escalation"] == "auto_handle"
                        and pr["intent_correct"]
                        and not flags
                    ):
                        flips["lost_safe"] += 1

        gold_intents = golden["intent"].astype(str).tolist()
        rk = recall_at_k(retrieved_intents, gold_intents)
        y_true = [pred_rows[eid]["gold_escalation"] for eid in golden["example_id"]]
        # align order with golden iteration
        y_true = []
        intent_ok = []
        flags_list = []
        for _, row in golden.iterrows():
            pr = pred_rows[row["example_id"]]
            y_true.append(pr["gold_escalation"])
            intent_ok.append(pr["intent_correct"])
            flags_list.append(pr.get("safety_flags") or [])

        em = escalation_metrics(y_true, e2e_preds)
        should = [i for i, g in enumerate(y_true) if g == "escalate"]
        fah = sum(1 for i in should if e2e_preds[i] == "auto_handle") / len(should)
        sah = sum(
            1
            for i, g in enumerate(y_true)
            if e2e_preds[i] == "auto_handle"
            and intent_ok[i]
            and not flags_list[i]
            and g == "auto_handle"
        ) / len(y_true)

        # FAH case table for baseline mode
        if mode == "none":
            for _, row in golden.iterrows():
                pr = pred_rows[row["example_id"]]
                if pr["gold_escalation"] == "escalate" and pr["predicted_escalation"] == "auto_handle":
                    fah_rows.append(
                        {
                            "example_id": pr["example_id"],
                            "gold_intent": pr["gold_intent"],
                            "predicted_intent": pr["predicted_intent"],
                            "intent_correct": pr["intent_correct"],
                            "top_ev_intents": [e.get("intent") for e in (pr.get("evidence") or [])],
                        }
                    )

        results.append(
            {
                "mode": mode,
                "recall_at_1": rk["recall_at_1"],
                "recall_at_3": rk["recall_at_3"],
                "recall_at_5": rk["recall_at_5"],
                "auto_handle_rate": em["auto_handle_rate"],
                "sah": sah,
                "fah_among_should": fah,
                "escalate_f1": em["escalate_f1"],
                "escalate_precision": em["escalate_precision"],
                "escalate_recall": em["escalate_recall"],
                "flips": flips,
            }
        )
        print(
            f"{mode:12} R@1={rk['recall_at_1']:.3f} R@3={rk['recall_at_3']:.3f} R@5={rk['recall_at_5']:.3f} "
            f"SAH={sah:.3f} auto={em['auto_handle_rate']:.3f} FAH={fah:.3f} F1={em['escalate_f1']:.3f} "
            f"flips={flips}"
        )

    out = {
        "policy": "calibrated_high_risk_includes_order_quality_issue",
        "candidate_pool": args.pool,
        "note": (
            "Rerank uses predicted_intent from frozen GPT predictions (inference-time). "
            "Gold intent never enters the score. End-to-end escalation recomputed with "
            "frozen intents/replies/safety_flags; only evidence/sufficiency changes."
        ),
        "variants": results,
        "fah_baseline_cases": fah_rows,
    }
    path = ROOT / "artifacts" / "final" / "retrieval_rerank_ablation.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
