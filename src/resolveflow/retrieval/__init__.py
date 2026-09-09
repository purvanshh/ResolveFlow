"""Retrieval index types and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RetrievedCase:
    case_id: str
    similarity: float
    customer_message: str
    brand_response: str
    intent: str
    resolution_summary: str
    customer_context: str = ""
    conversation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
