"""LLM provider abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema_hint: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return a JSON-compatible dict. Must not raise on empty content — return {}."""

    def generate_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> str:
        data = self.generate_structured(
            system=system,
            user=user + "\n\nReturn JSON: {\"text\": \"...\"}",
            schema_hint={"text": "string"},
            temperature=temperature,
        )
        return str(data.get("text") or data.get("reply") or "")
