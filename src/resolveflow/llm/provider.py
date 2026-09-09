"""OpenAI chat provider with JSON responses."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from resolveflow.llm.base import LLMProvider
from resolveflow.llm.mock import MockProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("RESOLVEFLOW_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAIProvider. "
                "Set it in the environment or use --mock / offline mode."
            )

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema_hint: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai: pip install openai") from exc

        client = OpenAI(api_key=self.api_key)
        hint = ""
        if schema_hint:
            hint = "\nRespond with a JSON object matching keys: " + ", ".join(
                schema_hint.keys()
            )
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system + hint},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        usage = getattr(resp, "usage", None)
        data = _parse_json(content)
        if usage is not None:
            data["_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        data["_model"] = self.model
        return data


def _parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"text": text}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def get_provider(mode: str = "auto", *, model: str | None = None, intents: list[str] | None = None) -> LLMProvider:
    """
    mode: auto | openai | mock
    auto uses OpenAI when OPENAI_API_KEY is set, else mock.
    """
    mode = (mode or "auto").lower()
    if mode == "mock":
        return MockProvider(intents=intents)
    if mode == "openai":
        return OpenAIProvider(model=model)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider(model=model)
    return MockProvider(intents=intents)
