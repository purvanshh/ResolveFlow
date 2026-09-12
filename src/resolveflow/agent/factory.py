"""Factory helpers to construct a configured AgentPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from resolveflow.agent.classifier import IntentClassifier
from resolveflow.agent.pipeline import AgentPipeline
from resolveflow.agent.policy import EscalationPolicy
from resolveflow.agent.responder import Responder
from resolveflow.baselines import TfidfBaseline
from resolveflow.config import load_config, resolve_path
from resolveflow.llm import get_provider
from resolveflow.retrieval.retrieve import load_retriever
from resolveflow.taxonomy import intent_names, load_taxonomy


def build_agent(
    *,
    config_path: str | Path | None = None,
    provider_mode: str = "auto",
    classifier_mode: str | None = None,
    responder_mode: str | None = None,
    top_k: int | None = None,
    retrieval_mode: str | None = None,
    similarity_threshold: float | None = None,
    disable_retrieval: bool = False,
    thinking_enabled: bool | None = None,
) -> tuple[AgentPipeline, dict[str, Any]]:
    cfg = load_config(config_path)
    tax = load_taxonomy()
    brand = cfg["brand"]["name"]
    agent_cfg = cfg.get("agent", {})
    esc_cfg = cfg.get("escalation", {})
    safety_cfg = cfg.get("safety", {})
    rcfg = cfg["retrieval"]

    import os

    from resolveflow.config import load_dotenv

    load_dotenv()

    # Config may pin provider (e.g. deepseek experiments)
    cfg_provider = (agent_cfg.get("provider") or "").lower()
    if provider_mode == "auto" and cfg_provider:
        provider_mode = cfg_provider

    think = (
        bool(thinking_enabled)
        if thinking_enabled is not None
        else bool(agent_cfg.get("thinking_enabled", False))
    )
    model = (
        agent_cfg.get("classifier_model")
        or os.getenv("RESOLVEFLOW_LLM_MODEL")
        or "gpt-4o-mini"
    )
    model_version = agent_cfg.get("model_version")
    reasoning_effort = str(agent_cfg.get("reasoning_effort") or "high")
    base_url = agent_cfg.get("base_url")

    has_openai = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    has_deepseek = bool((os.getenv("DEEPSEEK_API_KEY") or "").strip())

    if provider_mode in {"offline", "tfidf"}:
        classifier_mode = classifier_mode or "tfidf"
        responder_mode = responder_mode or "grounded_template"
        provider = get_provider("mock", intents=intent_names(tax))
    elif provider_mode == "deepseek":
        if not has_deepseek:
            raise RuntimeError("DEEPSEEK_API_KEY required for provider_mode=deepseek")
        provider = get_provider(
            "deepseek",
            model=model,
            thinking_enabled=think,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            model_version=model_version,
            intents=intent_names(tax),
        )
        classifier_mode = classifier_mode or agent_cfg.get("classifier_mode") or "llm"
        responder_mode = responder_mode or agent_cfg.get("responder_mode") or "llm"
    elif provider_mode in {"openai"} or (provider_mode == "auto" and has_openai):
        provider = get_provider("openai", model=model, intents=intent_names(tax))
        classifier_mode = classifier_mode or agent_cfg.get("classifier_mode") or "llm"
        responder_mode = responder_mode or agent_cfg.get("responder_mode") or "llm"
    elif provider_mode == "auto" and not has_openai:
        classifier_mode = classifier_mode or "tfidf"
        responder_mode = responder_mode or "grounded_template"
        provider = get_provider("mock", intents=intent_names(tax))
    else:
        provider = get_provider(
            provider_mode,
            model=model,
            thinking_enabled=think,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            model_version=model_version,
            intents=intent_names(tax),
        )
        classifier_mode = classifier_mode or agent_cfg.get("classifier_mode") or "llm"
        responder_mode = responder_mode or agent_cfg.get("responder_mode") or "llm"

    tfidf = None
    model_path = resolve_path(cfg, cfg["baseline"]["model_path"])
    if model_path.exists():
        tfidf = TfidfBaseline.load(model_path)

    classifier = IntentClassifier(
        provider,
        brand=brand,
        taxonomy=tax,
        tfidf=tfidf,
        mode=classifier_mode,
    )
    responder = Responder(
        provider,
        brand=brand,
        max_chars=int(safety_cfg.get("max_reply_characters", 280)),
        mode=responder_mode,
    )
    policy = EscalationPolicy(
        high_risk_intents=list(esc_cfg.get("high_risk_intents") or []),
        min_intent_confidence=float(esc_cfg.get("min_intent_confidence", 0.55)),
        require_evidence=bool(esc_cfg.get("require_evidence", True)),
        risk_threshold=int(esc_cfg.get("risk_threshold", 3)),
    )

    retriever = None
    if not disable_retrieval and top_k != 0:
        try:
            retriever = load_retriever(
                embeddings_path=resolve_path(cfg, rcfg["embeddings_path"]),
                metadata_path=resolve_path(cfg, rcfg["metadata_path"]),
                case_ids_path=resolve_path(cfg, rcfg["cases_id_map_path"]),
                cases_path=resolve_path(cfg, cfg["data"]["historical_cases"]),
            )
        except FileNotFoundError:
            retriever = None

    pipeline = AgentPipeline(
        classifier=classifier,
        retriever=retriever,
        responder=responder,
        policy=policy,
        brand=brand,
        top_k=int(top_k if top_k is not None else agent_cfg.get("top_k", rcfg.get("top_k", 3))),
        similarity_threshold=float(
            similarity_threshold
            if similarity_threshold is not None
            else agent_cfg.get("similarity_threshold", rcfg.get("similarity_threshold", 0.45))
        ),
        retrieval_mode=(retrieval_mode or agent_cfg.get("retrieval_mode", "auto")),  # type: ignore[arg-type]
        rerank_mode=str(rcfg.get("rerank_mode") or agent_cfg.get("rerank_mode") or "none"),
        candidate_pool=int(rcfg.get("candidate_pool") or agent_cfg.get("candidate_pool") or 20),
        max_reply_chars=int(safety_cfg.get("max_reply_characters", 280)),
        skip_generation_on_high_risk=bool(agent_cfg.get("skip_generation_on_high_risk", True)),
    )
    meta: dict[str, Any] = {
        "provider": provider.name,
        "classifier_mode": classifier_mode,
        "responder_mode": responder_mode,
        "brand": brand,
        "model": getattr(provider, "model", model),
        "thinking_enabled": think if provider.name == "deepseek" else False,
    }
    if hasattr(provider, "request_extras"):
        meta["request"] = provider.request_extras()  # type: ignore[attr-defined]
    if model_version:
        meta["model_version"] = model_version
    return pipeline, meta
