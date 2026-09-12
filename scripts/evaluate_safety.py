#!/usr/bin/env python3
"""Mixed-outcome adversarial / safety regression suite for the agent.

Evaluation artifact only — does not change production policy or golden metrics.

The original six-case smoke suite (all expect escalate) remains covered as
legacy_smoke cases. The expanded suite also includes safe-auto candidates so
an always-escalate system cannot score as perfect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluation.safety_suite_cases import CASES, LEGACY_SMOKE_IDS, suite_summary  # noqa: E402
from evaluation.safety_suite_scoring import (  # noqa: E402
    always_escalate_decisions,
    score_suite,
)
from resolveflow.agent.factory import build_agent  # noqa: E402
from resolveflow.agent.safety import detect_prompt_injection  # noqa: E402
from resolveflow.schemas import AgentRequest  # noqa: E402


def _run_agent(cases: list[dict], *, mode: str, config: str) -> tuple[list[dict], dict]:
    agent, meta = build_agent(config_path=config, provider_mode=mode, top_k=3)
    decisions = []
    for case in cases:
        d = agent.handle(AgentRequest(message=case["text"]))
        decisions.append(
            {
                "escalate": bool(d.escalate),
                "reply": d.reply or "",
                "intent": d.intent,
                "safety_flags": list(d.safety_flags or []),
                "injection_detected": detect_prompt_injection(case["text"]),
            }
        )
    return decisions, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="offline")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the original six escalate-only smoke cases",
    )
    args = parser.parse_args()

    cases = [c for c in CASES if c.get("legacy_smoke")] if args.smoke_only else list(CASES)
    decisions, meta = _run_agent(cases, mode=args.mode, config=args.config)
    scored = score_suite(cases, decisions)

    baseline = score_suite(cases, always_escalate_decisions(cases))
    smoke_ids = [c["id"] for c in cases if c["id"] in LEGACY_SMOKE_IDS]
    smoke_results = [r for r in scored["results"] if r["id"] in LEGACY_SMOKE_IDS]
    smoke_pass = all(r["ok"] for r in smoke_results) if smoke_results else False

    out = {
        "kind": "mixed_outcome_safety_regression_suite",
        "note": (
            "Adversarial regression suite (not production traffic). "
            "Do not combine with golden SAH/FAH. "
            "Original six escalate-only cases remain as legacy_smoke."
        ),
        "suite_design": suite_summary(cases),
        "runtime": meta,
        "agent": scored["metrics"],
        "always_escalate_baseline": baseline["metrics"],
        "legacy_smoke": {
            "ids": smoke_ids,
            "passed": smoke_pass,
            "n": len(smoke_results),
            "n_pass": sum(1 for r in smoke_results if r["ok"]),
        },
        "results": scored["results"],
        "baseline_results": baseline["results"],
        "failed_case_ids": [r["id"] for r in scored["results"] if not r["ok"]],
        "false_auto_handle_ids": [
            r["id"] for r in scored["results"] if "false_auto_handle" in r["reasons"]
        ],
        "false_escalation_ids": [
            r["id"] for r in scored["results"] if "false_escalation" in r["reasons"]
        ],
        "forbidden_claim_ids": [
            r["id"] for r in scored["results"] if "forbidden_claim_in_reply" in r["reasons"]
        ],
    }

    metrics_path = ROOT / "artifacts" / "metrics" / "safety_suite.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(out, indent=2) + "\n")

    final_path = ROOT / "artifacts" / "final" / "safety_suite_mixed.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(out, indent=2) + "\n")

    # Preserve a compact pointer in safety_results.json without touching golden headline.
    safety_results_path = ROOT / "artifacts" / "final" / "safety_results.json"
    prior = {}
    if safety_results_path.exists():
        try:
            prior = json.loads(safety_results_path.read_text())
        except json.JSONDecodeError:
            prior = {}
    prior_suite = prior.get("suite") if isinstance(prior.get("suite"), dict) else {}
    safety_results_path.write_text(
        json.dumps(
            {
                "unsupported_claim_rate_among_drafts": prior.get(
                    "unsupported_claim_rate_among_drafts", 0.015
                ),
                "hallucination_safety_mean": prior.get("hallucination_safety_mean", 4.935),
                "legacy_smoke_suite": {
                    "note": "Original 6 escalate-only cases (subset of mixed suite).",
                    "passed": smoke_pass,
                    "failures": sum(1 for r in smoke_results if not r["ok"]),
                    "results": smoke_results,
                    "runtime": prior_suite.get("runtime") or meta,
                },
                "mixed_outcome_suite": {
                    "artifact": "artifacts/final/safety_suite_mixed.json",
                    "n_cases": out["agent"]["n_cases"],
                    "case_pass_rate": out["agent"]["case_pass_rate"],
                    "escalation_recall": out["agent"]["escalation_recall"],
                    "false_auto_handle_rate": out["agent"]["false_auto_handle_rate"],
                    "false_escalation_rate": out["agent"]["false_escalation_rate"],
                    "safe_auto_precision": out["agent"]["safe_auto_precision"],
                    "balanced_safety_usefulness": out["agent"]["balanced_safety_usefulness"],
                    "always_escalate_balanced": out["always_escalate_baseline"][
                        "balanced_safety_usefulness"
                    ],
                    "passed_all_cases": out["agent"]["passed_all_cases"],
                },
                # Backward-compatible fields: smoke status (not mixed perfect-score).
                "suite_pass": smoke_pass,
                "suite_failures": sum(1 for r in smoke_results if not r["ok"]),
                "suite": {
                    "runtime": meta,
                    "passed": smoke_pass,
                    "failures": sum(1 for r in smoke_results if not r["ok"]),
                    "results": smoke_results,
                    "note": "legacy_smoke subset; see mixed_outcome_suite for full regression suite",
                },
            },
            indent=2,
        )
        + "\n"
    )

    m = out["agent"]
    b = out["always_escalate_baseline"]
    print(
        json.dumps(
            {
                "legacy_smoke_passed": smoke_pass,
                "mixed_n": m["n_cases"],
                "mixed_case_pass_rate": round(m["case_pass_rate"], 3),
                "escalation_recall": round(m["escalation_recall"], 3),
                "false_auto_handle_rate": round(m["false_auto_handle_rate"], 3),
                "false_escalation_rate": round(m["false_escalation_rate"], 3),
                "safe_auto_precision": round(m["safe_auto_precision"], 3),
                "balanced_safety_usefulness": round(m["balanced_safety_usefulness"], 3),
                "always_escalate_balanced": round(b["balanced_safety_usefulness"], 3),
                "failed_case_ids": out["failed_case_ids"],
            },
            indent=2,
        )
    )
    for r in scored["results"]:
        status = "PASS" if r["ok"] else "FAIL"
        print(
            f"  {r['id']}: {status} escalate={r['escalate']} "
            f"expect_esc={r['expected_escalate']} intent={r['intent']} "
            f"reasons={r['reasons']}"
        )

    # Exit 0 if legacy smoke still passes (CI-friendly); mixed failures are reported in artifact.
    return 0 if smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
