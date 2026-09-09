#!/usr/bin/env python3
"""Run the ResolveFlow agent on a single message."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.agent.factory import build_agent  # noqa: E402
from resolveflow.schemas import AgentRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Force mock LLM provider")
    parser.add_argument("--offline", action="store_true", help="TF-IDF + grounded template (no API)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    mode = "mock" if args.mock else ("offline" if args.offline else "auto")
    agent, meta = build_agent(
        config_path=args.config,
        provider_mode=mode,
        top_k=0 if args.no_retrieval else args.top_k,
        disable_retrieval=args.no_retrieval,
    )
    decision = agent.handle(AgentRequest(message=args.text, context=args.context or None))

    if args.json:
        payload = decision.to_dict()
        payload["_runtime"] = meta
        print(json.dumps(payload, indent=2))
        return 0

    print("ResolveFlow")
    print("===========")
    print()
    print(f"Brand:\n{meta['brand']}")
    print()
    print(f"Customer:\n{args.text}")
    print()
    print(f"Intent:\n{decision.intent}")
    print()
    print(f"Confidence:\n{decision.confidence:.2f}")
    print()
    print("Historical Evidence:")
    if not decision.evidence:
        print("(none)")
    else:
        for e in decision.evidence:
            print(f"  {e.case_id}   similarity={e.similarity:.2f}   intent={e.intent}")
    print()
    print("Safety:")
    print("PASS" if not decision.safety_flags else "FAIL " + ", ".join(decision.safety_flags))
    print()
    print("Decision:")
    print("ESCALATE" if decision.escalate else "AUTO-HANDLE")
    print()
    if decision.escalation_reason:
        print("Reason:")
        print(decision.escalation_reason)
        print()
    print("Draft:")
    print(decision.reply or "(no customer-facing draft; escalated)")
    print()
    print(f"Runtime: provider={meta['provider']} classifier={meta['classifier_mode']} responder={meta['responder_mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
