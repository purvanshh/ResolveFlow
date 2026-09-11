"""Risk-aware escalation policy (deterministic reasons)."""

from __future__ import annotations

from resolveflow.agent.safety import (
    ACCOUNT_SPECIFIC_CLAIM,
    MISSING_EVIDENCE,
    PROMPT_INJECTION_DETECTED,
    UNSUPPORTED_ACTION,
    UNSUPPORTED_MONETARY_CLAIM,
    UNSUPPORTED_POLICY,
    UNSUPPORTED_TIMELINE,
    detect_human_request,
)
from resolveflow.schemas import EscalationDecision, IntentPrediction, SafetyResult


class EscalationPolicy:
    def __init__(
        self,
        *,
        high_risk_intents: list[str] | None = None,
        min_intent_confidence: float = 0.55,
        require_evidence: bool = True,
        risk_threshold: int = 3,
    ):
        self.high_risk_intents = set(
            high_risk_intents
            or [
                "payment_billing",
                "refund_status",
                "refund_request",
                "account_access",
                "package_missing_or_misdelivered",
                "order_quality_issue",
            ]
        )
        self.min_intent_confidence = min_intent_confidence
        self.require_evidence = require_evidence
        self.risk_threshold = risk_threshold

    def decide(
        self,
        *,
        message: str,
        intent: IntentPrediction,
        top_similarity: float | None,
        evidence_count: int,
        evidence_sufficient: bool,
        safety: SafetyResult,
        skip_generation: bool = False,
    ) -> EscalationDecision:
        risk = 0
        triggers: list[str] = []

        if intent.intent == "other_unclear":
            risk += 2
            triggers.append("unclear_intent")

        if intent.confidence < self.min_intent_confidence:
            risk += 2
            triggers.append("low_intent_confidence")

        if intent.intent in self.high_risk_intents:
            risk += 3
            triggers.append("high_risk_intent")

        if self.require_evidence and not evidence_sufficient:
            risk += 2
            triggers.append("insufficient_evidence")

        if evidence_count == 0 or (top_similarity is not None and top_similarity <= 0):
            risk += 2
            if "insufficient_evidence" not in triggers:
                triggers.append("no_retrieval_hits")

        if detect_human_request(message):
            risk += 3
            triggers.append("explicit_human_request")

        severe_flags = {
            UNSUPPORTED_MONETARY_CLAIM,
            UNSUPPORTED_TIMELINE,
            UNSUPPORTED_ACTION,
            UNSUPPORTED_POLICY,
            ACCOUNT_SPECIFIC_CLAIM,
            PROMPT_INJECTION_DETECTED,
            MISSING_EVIDENCE,
        }
        if any(f in severe_flags for f in safety.flags):
            risk += 3
            triggers.append("safety_flags")

        if skip_generation:
            risk += 2
            triggers.append("skipped_generation_high_risk")

        escalate = risk >= self.risk_threshold or bool(
            set(triggers)
            & {
                "high_risk_intent",
                "explicit_human_request",
                "safety_flags",
                "unclear_intent",
                "insufficient_evidence",
                "no_retrieval_hits",
            }
        )

        # Allow auto-handle only when clean
        can_auto = (
            intent.intent != "other_unclear"
            and intent.confidence >= self.min_intent_confidence
            and intent.intent not in self.high_risk_intents
            and evidence_sufficient
            and safety.safe
            and not detect_human_request(message)
        )
        if can_auto:
            escalate = False
            triggers = ["eligible_auto_handle"]
            reason = None
        else:
            escalate = True
            reason = self._reason(triggers)

        return EscalationDecision(
            escalate=escalate,
            reason=reason,
            triggers=triggers,
            risk_score=risk,
        )

    def _reason(self, triggers: list[str]) -> str:
        if "safety_flags" in triggers:
            return "Escalated because the drafted response contained unsupported claims."
        if "explicit_human_request" in triggers:
            return "Escalated because the customer requested a human agent."
        if "high_risk_intent" in triggers:
            return "Escalated because this request falls into a high-risk support category."
        if "insufficient_evidence" in triggers or "no_retrieval_hits" in triggers:
            return "Escalated because no sufficiently similar historical support case was found."
        if "unclear_intent" in triggers:
            return (
                "Escalated because the message does not contain enough information "
                "to identify the customer's issue reliably."
            )
        if "low_intent_confidence" in triggers:
            return "Escalated because intent classification confidence was insufficient."
        return "Escalated because account-specific verification or review is required."
