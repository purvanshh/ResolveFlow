"""DeepSeek OpenAI-compatible chat provider with thinking mode support."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from resolveflow.llm.base import LLMProvider

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MODEL_VERSION = "DeepSeek-V4.1-Flash"


class DeepSeekProvider(LLMProvider):
    """
    DeepSeek Chat Completions via OpenAI-compatible SDK.

    Thinking mode uses the official API parameter:
      extra_body={"thinking": {"type": "enabled"|"disabled"}}
    Optional effort: reasoning_effort="high" when thinking is enabled.

    Internal reasoning_content is never returned to callers as customer text.
    """

    name = "deepseek"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        model_version: str = MODEL_VERSION,
        max_tokens: int | None = None,
    ):
        self.model = model or os.getenv("RESOLVEFLOW_DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip()
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.thinking_enabled = bool(thinking_enabled)
        self.reasoning_effort = reasoning_effort
        self.model_version = model_version
        # Thinking consumes completion budget; give headroom for JSON answer.
        if max_tokens is not None:
            self.max_tokens = max_tokens
        else:
            self.max_tokens = 2048 if self.thinking_enabled else 1024
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required for DeepSeekProvider. "
                "Set it in .env or use --mock / offline mode."
            )

    def request_extras(self) -> dict[str, Any]:
        """Safe metadata describing the API request shape (no secrets, no CoT)."""
        thinking_type = "enabled" if self.thinking_enabled else "disabled"
        extras: dict[str, Any] = {
            "provider": self.name,
            "model": self.model,
            "model_version": self.model_version,
            "base_url": self.base_url,
            "thinking_enabled": self.thinking_enabled,
            "thinking": {"type": thinking_type},
        }
        if self.thinking_enabled:
            extras["reasoning_effort"] = self.reasoning_effort
        return extras

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

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        hint = ""
        if schema_hint:
            hint = "\nRespond with a JSON object matching keys: " + ", ".join(
                schema_hint.keys()
            )
        messages = [
            {"role": "system", "content": system + hint},
            {"role": "user", "content": user},
        ]
        thinking_type = "enabled" if self.thinking_enabled else "disabled"
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": messages,
            "max_tokens": self.max_tokens,
            "extra_body": {"thinking": {"type": thinking_type}},
        }
        if self.thinking_enabled:
            create_kwargs["reasoning_effort"] = self.reasoning_effort

        last_err: Exception | None = None
        resp = None
        retries = 0
        t0 = time.perf_counter()
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(**create_kwargs)
                break
            except Exception as exc:  # noqa: BLE001 — retry transient API/network errors
                last_err = exc
                retries = attempt + 1
                time.sleep(min(2**attempt, 30))
        latency_s = time.perf_counter() - t0
        if resp is None:
            raise RuntimeError(
                f"DeepSeek request failed after retries: {last_err}"
            ) from last_err

        message = resp.choices[0].message
        # Prefer final answer content; never surface reasoning_content as reply.
        content = getattr(message, "content", None) or "{}"
        usage = getattr(resp, "usage", None)
        data = _parse_json(content)
        data["_model"] = self.model
        data["_model_version"] = self.model_version
        data["_provider"] = self.name
        data["_thinking_enabled"] = self.thinking_enabled
        data["_latency_s"] = latency_s
        data["_retries"] = retries
        data["_has_reasoning_content"] = bool(
            getattr(message, "reasoning_content", None)
        )
        if usage is not None:
            data["_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
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
