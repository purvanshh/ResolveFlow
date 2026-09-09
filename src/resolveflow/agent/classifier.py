"""Intent classification component."""

from __future__ import annotations

from typing import Any

from resolveflow.baselines import TfidfBaseline, combine_text
from resolveflow.llm.base import LLMProvider
from resolveflow.prompts.classifier import (
    build_classifier_system,
    build_classifier_user,
    taxonomy_block,
)
from resolveflow.prompts.versions import CLASSIFIER_VERSION
from resolveflow.schemas import IntentPrediction
from resolveflow.taxonomy import intent_names, load_taxonomy


class IntentClassifier:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        brand: str,
        taxonomy: dict[str, Any] | None = None,
        tfidf: TfidfBaseline | None = None,
        mode: str = "llm",  # llm | tfidf | hybrid
    ):
        self.provider = provider
        self.brand = brand
        self.taxonomy = taxonomy or load_taxonomy()
        self.allowed = set(intent_names(self.taxonomy))
        self.tfidf = tfidf
        self.mode = mode

    def classify(self, message: str, context: str = "") -> IntentPrediction:
        if self.mode == "tfidf":
            return self._tfidf_predict(message, context)
        if self.mode == "hybrid" and self.tfidf is not None:
            llm_pred = self._llm_predict(message, context)
            tf_intent, tf_conf = self.tfidf.predict_intent(message, context)
            if llm_pred.intent == "other_unclear" and tf_conf >= 0.55 and tf_intent in self.allowed:
                return IntentPrediction(
                    intent=tf_intent,
                    confidence=float(tf_conf),
                    reasoning_signals=["hybrid_tfidf_override_unclear"],
                )
            return llm_pred
        return self._llm_predict(message, context)

    def _tfidf_predict(self, message: str, context: str) -> IntentPrediction:
        if self.tfidf is None:
            return IntentPrediction(
                intent="other_unclear",
                confidence=0.0,
                reasoning_signals=["tfidf_unavailable"],
            )
        intent, conf = self.tfidf.predict_intent(message, context)
        if intent not in self.allowed:
            intent = "other_unclear"
            conf = min(conf, 0.3)
        return IntentPrediction(
            intent=intent,
            confidence=float(conf),
            reasoning_signals=["tfidf_classifier"],
        )

    def _llm_predict(self, message: str, context: str) -> IntentPrediction:
        system = build_classifier_system(self.brand)
        user = build_classifier_user(
            message=message,
            context=context,
            taxonomy_block=taxonomy_block(self.taxonomy),
        )
        data = self.provider.generate_structured(
            system=system,
            user=user,
            schema_hint={
                "intent": "string",
                "confidence": "number",
                "reasoning_signals": "array",
            },
            temperature=0.0,
        )
        intent = str(data.get("intent") or "").strip()
        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        signals = data.get("reasoning_signals") or []
        if not isinstance(signals, list):
            signals = [str(signals)]

        if intent not in self.allowed:
            # one retry with stricter instruction
            retry_user = user + f"\n\nPrevious invalid intent '{intent}'. Choose ONLY from: {sorted(self.allowed)}"
            data = self.provider.generate_structured(
                system=system,
                user=retry_user,
                schema_hint={"intent": "string", "confidence": "number"},
                temperature=0.0,
            )
            intent = str(data.get("intent") or "").strip()
            try:
                conf = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

        if intent not in self.allowed:
            return IntentPrediction(
                intent="other_unclear",
                confidence=0.0,
                reasoning_signals=["invalid_intent_fallback"],
                raw={"invalid": intent, "prompt_version": CLASSIFIER_VERSION},
            )
        return IntentPrediction(
            intent=intent,
            confidence=conf,
            reasoning_signals=[str(s) for s in signals][:5],
            raw={"prompt_version": CLASSIFIER_VERSION, "provider": self.provider.name},
        )
