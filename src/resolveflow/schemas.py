"""Typed contracts for ResolveFlow agent I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    message: str
    context: str | list[str] | None = None

    def context_text(self) -> str:
        if self.context is None:
            return ""
        if isinstance(self.context, list):
            return "\n".join(str(x) for x in self.context)
        return str(self.context)


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    reasoning_signals: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@dataclass
class RetrievedEvidence:
    case_id: str
    similarity: float
    customer_issue: str
    historical_response: str
    intent: str = ""
    resolution_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyResult:
    safe: bool
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str | None = None
    triggers: list[str] = field(default_factory=list)
    risk_score: int = 0


@dataclass
class AgentDecision:
    intent: str
    confidence: float
    reply: str
    escalate: bool
    escalation_reason: str | None
    evidence: list[RetrievedEvidence] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    reasoning_signals: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reply": self.reply,
            "escalate": self.escalate,
            "escalation_reason": self.escalation_reason,
            "evidence": [e.to_dict() for e in self.evidence],
            "safety_flags": list(self.safety_flags),
            "reasoning_signals": list(self.reasoning_signals),
            "meta": dict(self.meta),
        }
