#!/usr/bin/env python3
"""Streamlit UI for labeling / reviewing the golden evaluation set."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resolveflow.golden import (  # noqa: E402
    GOLDEN_COLUMNS,
    load_golden_set,
    save_golden_set,
    validate_golden_labels,
)
from resolveflow.taxonomy import intent_names, load_taxonomy  # noqa: E402

PROGRESS_PATH = ROOT / "data" / "golden" / ".label_progress.json"
GOLDEN_PATH = ROOT / "data" / "golden" / "golden_set.csv"
CANDIDATES_PATH = ROOT / "data" / "interim" / "golden_candidates.csv"


def _load_progress() -> int:
    if PROGRESS_PATH.exists():
        import json

        return int(json.loads(PROGRESS_PATH.read_text()).get("index", 0))
    return 0


def _save_progress(index: int) -> None:
    import json

    PROGRESS_PATH.write_text(json.dumps({"index": index}) + "\n")


def _ensure_frame() -> pd.DataFrame:
    df = load_golden_set(GOLDEN_PATH)
    if len(df) == 0 and CANDIDATES_PATH.exists():
        cand = pd.read_csv(CANDIDATES_PATH).fillna("")
        rows = []
        for i, r in cand.iterrows():
            rows.append(
                {
                    "example_id": r.get("example_id") or f"gold_{i+1:03d}",
                    "conversation_id": str(r["conversation_id"]),
                    "message_id": str(r["message_id"]),
                    "input_text": r["input_text"],
                    "context": r.get("context", ""),
                    "intent": "",
                    "escalation_expected": "",
                    "escalation_reason": "",
                    "difficulty": "",
                    "annotation_notes": "",
                }
            )
        df = pd.DataFrame(rows)[GOLDEN_COLUMNS]
        save_golden_set(df, GOLDEN_PATH)
    return df


def main() -> None:
    st.set_page_config(page_title="ResolveFlow Golden Labeler", layout="wide")
    st.title("ResolveFlow — Golden set labeling")
    st.caption(
        "Do not show model predictions here. Label the customer's primary intent."
    )

    taxonomy = load_taxonomy(ROOT / "configs" / "intents.yaml")
    intents = intent_names(taxonomy)
    df = _ensure_frame()

    if "idx" not in st.session_state:
        st.session_state.idx = _load_progress()
    idx = st.session_state.idx
    idx = max(0, min(idx, len(df) - 1)) if len(df) else 0

    st.write(f"Example **{idx + 1} / {len(df)}**")
    if not len(df):
        st.warning("No golden candidates found.")
        return

    row = df.iloc[idx]
    st.subheader("Customer message")
    st.write(row["input_text"])
    st.subheader("Conversation context")
    st.text(row["context"] or "(none)")

    intent = st.selectbox(
        "Intent",
        options=[""] + intents,
        index=([""] + intents).index(row["intent"])
        if row["intent"] in ([""] + intents)
        else 0,
    )
    escalation = st.radio(
        "Should this be escalated?",
        options=["", "auto_handle", "escalate", "uncertain"],
        index=["", "auto_handle", "escalate", "uncertain"].index(
            row["escalation_expected"]
        )
        if row["escalation_expected"] in ["", "auto_handle", "escalate", "uncertain"]
        else 0,
        horizontal=True,
    )
    difficulty = st.radio(
        "Difficulty",
        options=["", "easy", "medium", "hard"],
        index=["", "easy", "medium", "hard"].index(row["difficulty"])
        if row["difficulty"] in ["", "easy", "medium", "hard"]
        else 0,
        horizontal=True,
    )
    reason = st.text_input("Escalation reason (required if escalate)", row["escalation_reason"])
    notes = st.text_area("Notes", row["annotation_notes"])

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save & Next", type="primary"):
            if not intent or not escalation or not difficulty:
                st.error("Intent, escalation, and difficulty are required.")
            elif escalation == "escalate" and not str(reason).strip():
                st.error("Escalation reason required when escalate is selected.")
            else:
                df.loc[df.index[idx], "intent"] = intent
                df.loc[df.index[idx], "escalation_expected"] = escalation
                df.loc[df.index[idx], "escalation_reason"] = reason
                df.loc[df.index[idx], "difficulty"] = difficulty
                df.loc[df.index[idx], "annotation_notes"] = notes
                save_golden_set(df, GOLDEN_PATH)
                nxt = min(idx + 1, len(df) - 1)
                st.session_state.idx = nxt
                _save_progress(nxt)
                st.rerun()
    with c2:
        if st.button("Previous") and idx > 0:
            st.session_state.idx = idx - 1
            _save_progress(idx - 1)
            st.rerun()
    with c3:
        if st.button("Validate all"):
            errs = validate_golden_labels(df, taxonomy, require_complete=False)
            incomplete = df[
                (df["intent"] == "")
                | (df["escalation_expected"] == "")
                | (df["difficulty"] == "")
            ]
            st.write(f"Incomplete rows: {len(incomplete)}")
            st.write(f"Validation issues: {len(errs)}")
            if errs:
                st.code("\n".join(errs[:40]))


if __name__ == "__main__":
    main()
