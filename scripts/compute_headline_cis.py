#!/usr/bin/env python3
"""Bootstrap 95% CIs for frozen GPT headline metrics (no model rerun).

Uses the same resampling style as Macro-F1 bootstrap in evaluation/run_all.py:
  n_boot=500, seed=42, percentile [2.5, 97.5].

Source of truth: artifacts/evaluation/agent_predictions.jsonl
Writes: artifacts/final/headline_confidence_intervals.json

Point estimates are recomputed from predictions and must match headline.json;
this script does not alter headline values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.metrics import escalation_metrics  # noqa: E402

DEFAULT_PREDS = ROOT / "artifacts/evaluation/agent_predictions.jsonl"
DEFAULT_OUT = ROOT / "artifacts/final/headline_confidence_intervals.json"
N_BOOT = 500
SEED = 42


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_safe_auto(r: dict) -> bool:
    return (
        r["predicted_escalation"] == "auto_handle"
        and bool(r.get("intent_correct"))
        and not (r.get("safety_flags") or [])
        and r["gold_escalation"] == "auto_handle"
    )


def point_metrics(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    sah = sum(1 for r in rows if is_safe_auto(r)) / n
    auto = sum(1 for r in rows if r["predicted_escalation"] == "auto_handle") / n
    should = [r for r in rows if r["gold_escalation"] == "escalate"]
    fah = (
        sum(1 for r in should if r["predicted_escalation"] == "auto_handle") / len(should)
        if should
        else float("nan")
    )
    esc = escalation_metrics(
        [r["gold_escalation"] for r in rows],
        [r["predicted_escalation"] for r in rows],
    )["escalate_f1"]
    return {
        "safe_auto_handling_rate": float(sah),
        "auto_handle_rate": float(auto),
        "false_auto_handle_rate_among_should_escalate": float(fah),
        "escalation_f1": float(esc),
    }


def bootstrap_ci(
    rows: list[dict],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    n = len(rows)
    keys = [
        "safe_auto_handling_rate",
        "auto_handle_rate",
        "false_auto_handle_rate_among_should_escalate",
        "escalation_f1",
    ]
    scores: dict[str, list[float]] = {k: [] for k in keys}
    skipped_fah = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [rows[i] for i in idx]
        m = point_metrics(sample)
        for k in keys:
            if k == "false_auto_handle_rate_among_should_escalate" and np.isnan(m[k]):
                skipped_fah += 1
                continue
            scores[k].append(m[k])

    out: dict[str, dict] = {}
    for k in keys:
        arr = np.asarray(scores[k], dtype=float)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[k] = {
            "bootstrap_mean": float(np.mean(arr)),
            "ci95": [float(lo), float(hi)],
            "n_boot_used": int(len(arr)),
        }
    out["false_auto_handle_rate_among_should_escalate"]["n_boot_skipped_empty_should_escalate"] = (
        skipped_fah
    )
    return out


def binomial_exact_ci(k: int, n: int, confidence: float = 0.95) -> list[float] | None:
    """Clopper–Pearson exact binomial CI when SciPy is available."""
    if n <= 0:
        return None
    try:
        from scipy.stats import binomtest
    except ImportError:
        return None
    interval = binomtest(k, n).proportion_ci(confidence_level=confidence, method="exact")
    return [float(interval.low), float(interval.high)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    if len(rows) != 200:
        raise SystemExit(f"Expected 200 frozen predictions, got {len(rows)}")

    points = point_metrics(rows)
    boot = bootstrap_ci(rows, n_boot=args.n_boot, seed=args.seed)

    n = len(rows)
    sah_k = sum(1 for r in rows if is_safe_auto(r))
    auto_k = sum(1 for r in rows if r["predicted_escalation"] == "auto_handle")
    should = [r for r in rows if r["gold_escalation"] == "escalate"]
    fah_k = sum(1 for r in should if r["predicted_escalation"] == "auto_handle")

    binomial = {
        "safe_auto_handling_rate": {
            "k": sah_k,
            "n": n,
            "ci95_exact": binomial_exact_ci(sah_k, n),
        },
        "auto_handle_rate": {
            "k": auto_k,
            "n": n,
            "ci95_exact": binomial_exact_ci(auto_k, n),
        },
        "false_auto_handle_rate_among_should_escalate": {
            "k": fah_k,
            "n": len(should),
            "ci95_exact": binomial_exact_ci(fah_k, len(should)),
            "note": "Conditioned on gold should-escalate subset size in the frozen set (115).",
        },
    }

    artifact = {
        "source_predictions": str(args.predictions.relative_to(ROOT)),
        "n_examples": n,
        "method": {
            "bootstrap": {
                "n_boot": args.n_boot,
                "seed": args.seed,
                "resample": "with_replacement over golden examples",
                "ci": "percentile [2.5, 97.5]",
                "consistent_with": "evaluation/run_all.py bootstrap_macro_f1",
            },
            "binomial_exact": {
                "method": "Clopper-Pearson via scipy.stats.binomtest (when scipy available)",
                "applies_to": "binary rates only (not Escalation F1)",
            },
        },
        "interpretation": (
            "Intervals quantify sampling variability over the frozen n=200 golden set under "
            "resampling. They do not correct for policy selection on the same set, judge noise, "
            "or domain shift. Point estimates are unchanged from headline.json."
        ),
        "point_estimates": points,
        "bootstrap_ci95": {
            "safe_auto_handling_rate": boot["safe_auto_handling_rate"],
            "auto_handle_rate": boot["auto_handle_rate"],
            "false_auto_handle_rate_among_should_escalate": boot[
                "false_auto_handle_rate_among_should_escalate"
            ],
            "escalation_f1": boot["escalation_f1"],
        },
        "binomial_exact_ci95": binomial,
        "display": {
            "Safe Auto-Handling": {
                "point": points["safe_auto_handling_rate"],
                "bootstrap_ci95": boot["safe_auto_handling_rate"]["ci95"],
            },
            "Auto-handle": {
                "point": points["auto_handle_rate"],
                "bootstrap_ci95": boot["auto_handle_rate"]["ci95"],
            },
            "FAH": {
                "point": points["false_auto_handle_rate_among_should_escalate"],
                "bootstrap_ci95": boot["false_auto_handle_rate_among_should_escalate"]["ci95"],
            },
            "Escalation F1": {
                "point": points["escalation_f1"],
                "bootstrap_ci95": boot["escalation_f1"]["ci95"],
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact["display"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
