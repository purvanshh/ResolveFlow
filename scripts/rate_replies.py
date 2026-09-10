#!/usr/bin/env python3
"""Streamlit UI for human reply ratings (calibration set)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

JUDGE_INPUTS = ROOT / "artifacts" / "final" / "judge_inputs.jsonl"
OUT = ROOT / "artifacts" / "final" / "human_ratings_streamlit.jsonl"
PROGRESS = ROOT / "artifacts" / "final" / ".rate_progress.json"


def load_inputs():
    if not JUDGE_INPUTS.exists():
        return []
    return [json.loads(l) for l in JUDGE_INPUTS.read_text().splitlines() if l.strip()]


def load_saved():
    if not OUT.exists():
        return {}
    out = {}
    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["example_id"]] = row
    return out


def main() -> None:
    st.set_page_config(page_title="ResolveFlow Reply Rater", layout="wide")
    st.title("ResolveFlow — Human reply rating")
    st.caption("Do not show LLM judge scores while rating.")

    rows = load_inputs()
    if not rows:
        st.error("Run `python -m evaluation.run_all` first to create judge_inputs.jsonl")
        return
    saved = load_saved()
    if "idx" not in st.session_state:
        st.session_state.idx = 0
        if PROGRESS.exists():
            st.session_state.idx = json.loads(PROGRESS.read_text()).get("index", 0)

    idx = max(0, min(st.session_state.idx, len(rows) - 1))
    row = rows[idx]
    st.write(f"Example **{idx+1} / {len(rows)}** (`{row['example_id']}`)")
    st.subheader("Customer")
    st.write(row["customer_message"])
    st.subheader("Evidence")
    st.json(row.get("evidence") or [])
    st.subheader("Agent reply")
    st.write(row.get("reply") or "(empty)")
    st.write(f"Escalate: {row.get('escalate')} — {row.get('escalation_reason')}")

    dims = [
        "correctness",
        "groundedness",
        "helpfulness",
        "completeness",
        "tone",
        "hallucination_safety",
        "escalation_appropriateness",
    ]
    scores = {}
    prev = saved.get(row["example_id"], {})
    cols = st.columns(4)
    for i, d in enumerate(dims):
        with cols[i % 4]:
            scores[d] = st.radio(
                d,
                [1, 2, 3, 4, 5],
                index=int(prev.get(d, 3)) - 1,
                key=f"{row['example_id']}_{d}",
                horizontal=True,
            )
    notes = st.text_area("Notes", prev.get("notes", ""))

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save & Next", type="primary"):
            rec = {
                "example_id": row["example_id"],
                "annotator": "human_streamlit",
                **scores,
                "overall": int(round(sum(scores.values()) / len(scores))),
                "notes": notes,
            }
            # append/replace
            saved[row["example_id"]] = rec
            with OUT.open("w") as f:
                for v in saved.values():
                    f.write(json.dumps(v) + "\n")
            nxt = min(idx + 1, len(rows) - 1)
            st.session_state.idx = nxt
            PROGRESS.write_text(json.dumps({"index": nxt}))
            st.rerun()
    with c2:
        if st.button("Previous") and idx > 0:
            st.session_state.idx = idx - 1
            PROGRESS.write_text(json.dumps({"index": idx - 1}))
            st.rerun()


if __name__ == "__main__":
    main()
