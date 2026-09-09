"""Format retrieved cases into compact LLM evidence blocks."""

from __future__ import annotations


def format_evidence(cases: list, *, max_chars_per_field: int = 280) -> str:
    if not cases:
        return "No sufficiently similar historical case found."

    blocks: list[str] = []
    for i, c in enumerate(cases, start=1):
        cust = _clip(
            getattr(c, "customer_message", "") or getattr(c, "customer_issue", ""),
            max_chars_per_field,
        )
        summary = _clip(
            getattr(c, "resolution_summary", None)
            or getattr(c, "brand_response", None)
            or getattr(c, "historical_response", ""),
            max_chars_per_field,
        )
        intent = getattr(c, "intent", "")
        sim = float(getattr(c, "similarity", 0.0))
        blocks.append(
            "\n".join(
                [
                    f"CASE {i}",
                    f"Customer issue:\n{cust}",
                    f"Historical response:\n{summary}",
                    f"Intent:\n{intent}",
                    f"Similarity:\n{sim:.3f}",
                ]
            )
        )
    return "\n\n".join(blocks)


def is_sufficient_evidence(
    results: list,
    *,
    similarity_threshold: float = 0.45,
    min_results: int = 1,
) -> bool:
    """Evidence is sufficient when ≥min_results cases clear the similarity floor."""
    if not results or len(results) < min_results:
        return False
    top = results[0]
    sim = float(getattr(top, "similarity", 0.0))
    return sim >= similarity_threshold


def similarity_gap(results: list) -> float | None:
    if len(results) < 2:
        return None
    return float(results[0].similarity) - float(results[1].similarity)


def _clip(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    if len(t) <= n:
        return t
    return t[: n - 3].rstrip() + "..."
