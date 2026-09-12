"""End-to-end ResolveFlow agent pipeline."""

from __future__ import annotations

import time
from typing import Any, Literal

from resolveflow.agent.classifier import IntentClassifier
from resolveflow.agent.policy import EscalationPolicy
from resolveflow.agent.responder import Responder
from resolveflow.agent.safety import (
    SafetyResult,
    detect_prompt_injection,
    validate_reply,
)
from resolveflow.prompts.versions import CLASSIFIER_VERSION, RESPONDER_VERSION
from resolveflow.retrieval.evidence import is_sufficient_evidence, similarity_gap
from resolveflow.retrieval.rerank import rerank_cases
from resolveflow.retrieval.retrieve import Retriever
from resolveflow.schemas import (
    AgentDecision,
    AgentRequest,
    RetrievedEvidence,
)

RetrievalMode = Literal["filtered", "global", "auto"]
RerankMode = Literal["none", "intent", "resolution", "combined"]


class AgentPipeline:
    def __init__(
        self,
        *,
        classifier: IntentClassifier,
        retriever: Retriever | None,
        responder: Responder,
        policy: EscalationPolicy,
        brand: str,
        top_k: int = 3,
        similarity_threshold: float = 0.45,
        retrieval_mode: RetrievalMode = "auto",
        rerank_mode: RerankMode | str = "none",
        candidate_pool: int = 20,
        max_reply_chars: int = 280,
        skip_generation_on_high_risk: bool = True,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.responder = responder
        self.policy = policy
        self.brand = brand
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.retrieval_mode = retrieval_mode
        self.rerank_mode = (rerank_mode or "none").lower()
        self.candidate_pool = max(int(candidate_pool), top_k)
        self.max_reply_chars = max_reply_chars
        self.skip_generation_on_high_risk = skip_generation_on_high_risk

    def handle(self, request: AgentRequest) -> AgentDecision:
        t0 = time.perf_counter()
        message = (request.message or "").strip()
        if not message:
            return AgentDecision(
                intent="other_unclear",
                confidence=0.0,
                reply="",
                escalate=True,
                escalation_reason="Escalated because the message is empty.",
                meta={"error": "empty_message"},
            )

        context = request.context_text()
        t_cls0 = time.perf_counter()
        # Ultra-thin messages without useful context → force abstain
        if len(message) <= 20 and len(context.strip()) < 40:
            from resolveflow.schemas import IntentPrediction

            intent = IntentPrediction(
                intent="other_unclear",
                confidence=0.2,
                reasoning_signals=["thin_message_without_context"],
            )
        else:
            intent = self.classifier.classify(message, context)
        t_cls = time.perf_counter() - t_cls0

        t_ret0 = time.perf_counter()
        evidence_objs, retrieval_meta = self._retrieve(message, context, intent.intent)
        t_ret = time.perf_counter() - t_ret0
        top_sim = evidence_objs[0].similarity if evidence_objs else None
        sufficient = is_sufficient_evidence(
            evidence_objs, similarity_threshold=self.similarity_threshold
        )

        obvious_high_risk = (
            intent.intent in self.policy.high_risk_intents
            or intent.intent == "other_unclear"
            or detect_prompt_injection(message)
        )
        skip_gen = self.skip_generation_on_high_risk and obvious_high_risk and not sufficient

        t_gen0 = time.perf_counter()
        if skip_gen:
            reply = ""
        else:
            try:
                reply = self.responder.draft(
                    message=message,
                    context=context,
                    intent=intent.intent,
                    evidence=evidence_objs,
                )
            except Exception as exc:  # noqa: BLE001
                reply = ""
                retrieval_meta["generation_error"] = str(exc)
        t_gen = time.perf_counter() - t_gen0

        evidence_text = "\n".join(
            f"{e.customer_issue}\n{e.historical_response}" for e in evidence_objs
        )
        if reply:
            safety = validate_reply(
                reply,
                customer_message=message,
                evidence_text=evidence_text,
                max_chars=self.max_reply_chars,
                require_evidence=self.policy.require_evidence,
                has_evidence=sufficient,
            )
        else:
            safety = SafetyResult(safe=True, flags=[], notes=["generation_skipped"])

        decision = self.policy.decide(
            message=message,
            intent=intent,
            top_similarity=top_sim,
            evidence_count=len(evidence_objs),
            evidence_sufficient=sufficient,
            safety=safety,
            skip_generation=skip_gen,
        )

        # If escalating due to safety on a drafted reply, keep draft for human assist
        return AgentDecision(
            intent=intent.intent,
            confidence=float(intent.confidence),
            reply=reply,
            escalate=decision.escalate,
            escalation_reason=decision.reason,
            evidence=evidence_objs,
            safety_flags=list(safety.flags),
            reasoning_signals=list(intent.reasoning_signals),
            meta={
                "brand": self.brand,
                "retrieval": retrieval_meta,
                "similarity_gap": similarity_gap(evidence_objs),
                "evidence_sufficient": sufficient,
                "policy_triggers": decision.triggers,
                "risk_score": decision.risk_score,
                "prompt_versions": {
                    "classifier": CLASSIFIER_VERSION,
                    "responder": RESPONDER_VERSION,
                },
                "latency_s": {
                    "classify": round(t_cls, 4),
                    "retrieve": round(t_ret, 4),
                    "generate": round(t_gen, 4),
                    "total": round(time.perf_counter() - t0, 4),
                },
                "provider": self.classifier.provider.name,
                "classifier_mode": self.classifier.mode,
                "responder_mode": self.responder.mode,
            },
        )

    def _retrieve(
        self, message: str, context: str, intent: str
    ) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
        meta: dict[str, Any] = {
            "mode": self.retrieval_mode,
            "top_k": self.top_k,
            "rerank_mode": self.rerank_mode,
            "candidate_pool": self.candidate_pool,
        }
        if self.retriever is None or self.top_k <= 0:
            meta["mode"] = "disabled"
            return [], meta

        mode = self.retrieval_mode
        use_rerank = self.rerank_mode not in {"none", "off", ""}

        # Baseline path: preserve historical filtered/global behavior exactly.
        if not use_rerank:
            hits = []
            if mode in {"filtered", "auto"}:
                hits = self.retriever.retrieve(
                    message,
                    context=context,
                    top_k=max(self.top_k * 4, 8),
                    similarity_threshold=0.0,
                )
                filtered = [h for h in hits if h.intent == intent]
                if len(filtered) >= 1:
                    hits = filtered[: self.top_k]
                    hits = [h for h in hits if h.similarity >= self.similarity_threshold]
                    meta["mode_used"] = "filtered"
                elif mode == "filtered":
                    hits = []
                    meta["mode_used"] = "filtered_empty"
                else:
                    hits = self.retriever.retrieve(
                        message,
                        context=context,
                        top_k=self.top_k,
                        similarity_threshold=self.similarity_threshold,
                    )
                    meta["mode_used"] = "global_fallback"
            else:
                hits = self.retriever.retrieve(
                    message,
                    context=context,
                    top_k=self.top_k,
                    similarity_threshold=self.similarity_threshold,
                )
                meta["mode_used"] = "global"
        else:
            pool = max(self.candidate_pool, self.top_k * 4, 8)
            hits = self.retriever.retrieve(
                message,
                context=context,
                top_k=pool,
                similarity_threshold=0.0,
            )
            query_for_rerank = f"{message}\n{context}".strip()
            hits = rerank_cases(
                hits,
                query=query_for_rerank,
                predicted_intent=intent,
                mode=self.rerank_mode,
                top_k=None,
            )
            hits = [h for h in hits if h.similarity >= self.similarity_threshold][
                : self.top_k
            ]
            meta["mode_used"] = f"pool_then_{self.rerank_mode}"

        evidence = [
            RetrievedEvidence(
                case_id=h.case_id,
                similarity=float(h.similarity),
                customer_issue=h.customer_message,
                historical_response=h.brand_response,
                intent=h.intent,
                resolution_summary=h.resolution_summary,
            )
            for h in hits[: self.top_k]
        ]
        meta["n_hits"] = len(evidence)
        if evidence:
            meta["top_similarity"] = evidence[0].similarity
            meta["top_intent"] = evidence[0].intent
        return evidence, meta
