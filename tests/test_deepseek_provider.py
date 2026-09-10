"""Offline tests for DeepSeek provider configuration (no live API)."""

from __future__ import annotations

import pytest

from resolveflow.llm.deepseek import DEFAULT_BASE_URL, DEFAULT_MODEL, MODEL_VERSION, DeepSeekProvider
from resolveflow.llm.provider import get_provider


def test_deepseek_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekProvider(model="deepseek-flash", thinking_enabled=False)


def test_deepseek_nonthinking_request_extras(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
    p = DeepSeekProvider(model="deepseek-flash", thinking_enabled=False)
    extras = p.request_extras()
    assert p.model == "deepseek-flash"
    assert p.model_version == MODEL_VERSION
    assert p.base_url == DEFAULT_BASE_URL
    assert extras["thinking_enabled"] is False
    assert extras["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in extras


def test_deepseek_thinking_request_extras(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
    p = DeepSeekProvider(
        model="deepseek-flash",
        thinking_enabled=True,
        reasoning_effort="high",
    )
    extras = p.request_extras()
    assert extras["thinking_enabled"] is True
    assert extras["thinking"] == {"type": "enabled"}
    assert extras["reasoning_effort"] == "high"
    assert extras["model"] == DEFAULT_MODEL
    assert extras["model_version"] == "DeepSeek-V4.1-Flash"


def test_get_provider_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
    p = get_provider("deepseek", model="deepseek-flash", thinking_enabled=True)
    assert p.name == "deepseek"
    assert p.thinking_enabled is True


def test_deepseek_generate_structured_payload(monkeypatch):
    """Verify create() kwargs include thinking toggle; do not call live API."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
    captured = {}

    class FakeMessage:
        content = '{"intent":"delivery_delay","confidence":0.9}'
        reasoning_content = "SECRET_CHAIN_OF_THOUGHT"

    class FakeChoice:
        message = FakeMessage()

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class FakeResp:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, *a, **k):
            captured["client_kwargs"] = k
            self.chat = FakeChat()

    import resolveflow.llm.deepseek as ds

    monkeypatch.setattr(ds, "OpenAI", FakeClient, raising=False)

    # Patch import path used inside generate_structured
    import sys
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    p = DeepSeekProvider(model="deepseek-flash", thinking_enabled=True)
    out = p.generate_structured(system="sys", user="user", schema_hint={"intent": "str"})
    assert captured["model"] == "deepseek-flash"
    assert captured["extra_body"]["thinking"]["type"] == "enabled"
    assert captured["reasoning_effort"] == "high"
    assert captured["response_format"] == {"type": "json_object"}
    assert out["intent"] == "delivery_delay"
    assert "SECRET_CHAIN_OF_THOUGHT" not in str(out)
    assert out["_thinking_enabled"] is True
    assert out["_has_reasoning_content"] is True
    assert out.get("reasoning_content") is None


def test_deepseek_nonthinking_payload(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
    captured = {}

    class FakeMessage:
        content = '{"reply":"ok"}'
        reasoning_content = None

    class FakeChoice:
        message = FakeMessage()

    class FakeResp:
        choices = [FakeChoice()]
        usage = None

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, *a, **k):
            self.chat = FakeChat()

    import sys
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    p = DeepSeekProvider(model="deepseek-flash", thinking_enabled=False)
    out = p.generate_structured(system="sys", user="user")
    assert captured["extra_body"]["thinking"]["type"] == "disabled"
    assert "reasoning_effort" not in captured
    assert out["reply"] == "ok"


def test_experiment_configs_exist():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    a = root / "configs/experiments/deepseek_v41_flash_nonthinking.yaml"
    b = root / "configs/experiments/deepseek_v41_flash_thinking.yaml"
    assert a.exists() and b.exists()
    from resolveflow.config import load_config

    ca = load_config(a)
    cb = load_config(b)
    assert ca["agent"]["thinking_enabled"] is False
    assert cb["agent"]["thinking_enabled"] is True
    assert ca["agent"]["classifier_model"] == "deepseek-flash"
    assert cb["agent"]["model_version"] == "DeepSeek-V4.1-Flash"
    assert ca["artifacts"]["final_dir"] != "artifacts/final"
    assert "deepseek_v41_flash_nonthinking" in ca["artifacts"]["final_dir"]
    assert ca["judge"]["model"] == "gpt-4o-mini"
