"""Format retrieved cases into compact LLM evidence blocks (Phase 4 prep)."""

from __future__ import annotations

from resolveflow.retrieval import RetrievedCase


def format_evidence(cases: list[RetrievedCase], *, max_chars_per_field: int = 280) -> str:
    if not cases:
        return "No sufficiently similar historical case found."

    blocks: list[str] = []
    for i, c in enumerate(cases, start=1):
        cust = _clip(c.customer_message, max_chars_per_field)
        summary = _clip(c.resolution_summary or c.brand_response, max_chars_per_field)
        blocks.append(
            "\n".join(
                [
                    f"CASE {i}",
                    f"Customer issue:\n{cust}",
                    f"Historical response:\n{summary}",
                    f"Intent:\n{c.intent}",
                    f"Similarity:\n{c.similarity:.3f}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _clip(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    if len(t) <= n:
        return t
    return t[: n - 3].rstrip() + "..."
