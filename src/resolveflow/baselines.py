"""Intent / reply / escalation baselines with a shared interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

EscalationDecision = Literal["auto_handle", "escalate", "uncertain"]


@dataclass
class Prediction:
    intent: str
    confidence: float
    escalation: EscalationDecision
    reply: str
    meta: dict[str, Any] | None = None


class Baseline:
    """Shared evaluation interface for Phase 3 baselines."""

    name: str = "baseline"

    def predict_intent(self, message: str, context: str | None = None) -> tuple[str, float]:
        raise NotImplementedError

    def draft_reply(self, message: str, context: str | None = None) -> str:
        raise NotImplementedError

    def decide_escalation(
        self, message: str, context: str | None = None
    ) -> EscalationDecision:
        raise NotImplementedError

    def predict(self, message: str, context: str | None = None) -> Prediction:
        intent, conf = self.predict_intent(message, context)
        return Prediction(
            intent=intent,
            confidence=conf,
            escalation=self.decide_escalation(message, context),
            reply=self.draft_reply(message, context),
        )


def combine_text(message: str, context: str | None = None) -> str:
    msg = (message or "").strip()
    ctx = (context or "").strip()
    if not ctx:
        return msg
    return f"{msg}\n\nContext:\n{ctx}"


class MajorityBaseline(Baseline):
    name = "majority"

    def __init__(self, majority_intent: str, trivial_reply: str):
        self.majority_intent = majority_intent
        self.trivial_reply = trivial_reply

    def predict_intent(self, message: str, context: str | None = None) -> tuple[str, float]:
        return self.majority_intent, 1.0

    def draft_reply(self, message: str, context: str | None = None) -> str:
        return self.trivial_reply

    def decide_escalation(
        self, message: str, context: str | None = None
    ) -> EscalationDecision:
        return "escalate"


class AlwaysEscalateBaseline(Baseline):
    name = "always_escalate"

    def __init__(self, majority_intent: str, trivial_reply: str):
        self.majority_intent = majority_intent
        self.trivial_reply = trivial_reply

    def predict_intent(self, message: str, context: str | None = None) -> tuple[str, float]:
        return self.majority_intent, 0.0

    def draft_reply(self, message: str, context: str | None = None) -> str:
        return self.trivial_reply

    def decide_escalation(
        self, message: str, context: str | None = None
    ) -> EscalationDecision:
        return "escalate"


class TfidfBaseline(Baseline):
    name = "tfidf_logistic_regression"

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        trivial_reply: str,
        confidence_threshold: float = 0.7,
    ):
        self.pipeline = pipeline
        self.trivial_reply = trivial_reply
        self.confidence_threshold = confidence_threshold

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[str],
        *,
        max_features: int = 30000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 2.0,
        max_iter: int = 1000,
        trivial_reply: str = "",
        confidence_threshold: float = 0.7,
    ) -> "TfidfBaseline":
        pipe = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        min_df=min_df,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipe.fit(texts, labels)
        return cls(
            pipe, trivial_reply=trivial_reply, confidence_threshold=confidence_threshold
        )

    def predict_intent(self, message: str, context: str | None = None) -> tuple[str, float]:
        text = combine_text(message, context)
        proba = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        label = str(self.pipeline.classes_[idx])
        return label, float(proba[idx])

    def draft_reply(self, message: str, context: str | None = None) -> str:
        return self.trivial_reply

    def decide_escalation(
        self, message: str, context: str | None = None
    ) -> EscalationDecision:
        _, conf = self.predict_intent(message, context)
        if conf < self.confidence_threshold:
            return "escalate"
        return "auto_handle"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "trivial_reply": self.trivial_reply,
                "confidence_threshold": self.confidence_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TfidfBaseline":
        data = joblib.load(path)
        return cls(
            data["pipeline"],
            trivial_reply=data["trivial_reply"],
            confidence_threshold=float(data["confidence_threshold"]),
        )


class NearestNeighborBaseline(Baseline):
    """Copy the brand response from the top retrieved historical case."""

    name = "nearest_neighbor_response"

    def __init__(
        self,
        retrieve_fn,
        *,
        trivial_reply: str,
        intent_fallback: str = "other_unclear",
        similarity_threshold: float = 0.45,
    ):
        self.retrieve_fn = retrieve_fn
        self.trivial_reply = trivial_reply
        self.intent_fallback = intent_fallback
        self.similarity_threshold = similarity_threshold

    def _top(self, message: str, context: str | None = None):
        hits = self.retrieve_fn(
            combine_text(message, context),
            top_k=1,
            similarity_threshold=self.similarity_threshold,
        )
        return hits[0] if hits else None

    def predict_intent(self, message: str, context: str | None = None) -> tuple[str, float]:
        hit = self._top(message, context)
        if hit is None:
            return self.intent_fallback, 0.0
        return str(hit.intent or self.intent_fallback), float(hit.similarity)

    def draft_reply(self, message: str, context: str | None = None) -> str:
        hit = self._top(message, context)
        if hit is None or not hit.brand_response:
            return self.trivial_reply
        return str(hit.brand_response)

    def decide_escalation(
        self, message: str, context: str | None = None
    ) -> EscalationDecision:
        hit = self._top(message, context)
        if hit is None:
            return "escalate"
        return "auto_handle"


def majority_label(labels: pd.Series | list[str]) -> str:
    s = pd.Series(list(labels))
    return str(s.value_counts().index[0])
