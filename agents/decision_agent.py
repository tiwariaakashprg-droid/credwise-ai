from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DecisionPackage:
    risk_summary: dict[str, Any]
    model_evidence: list[dict[str, Any]]
    policy_evidence: list[dict[str, Any]]
    analytics_context: dict[str, Any]
    recommendation: str
    human_review_required: bool
    disclaimer: str = "Educational decision-support prototype; a human remains the final decision authority."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionIntelligenceAgent:
    name = "decision_agent"

    def __init__(self, llm=None):
        self.llm = llm

    def build_package(self, prediction: dict, risk_raising: list[dict], risk_reducing: list[dict], policy_evidence: list[dict], analytics: dict | None = None) -> DecisionPackage:
        top_model = []
        for item in risk_raising[:5]:
            top_model.append({"feature": item.get("Feature"), "value": item.get("Feature_Value"), "shap_value": item.get("SHAP_Value"), "direction": "risk_raising"})
        for item in risk_reducing[:5]:
            top_model.append({"feature": item.get("Feature"), "value": item.get("Feature_Value"), "shap_value": item.get("SHAP_Value"), "direction": "risk_reducing"})
        required = prediction.get("risk_level") == "High Risk"
        return DecisionPackage(
            risk_summary=prediction,
            model_evidence=top_model,
            policy_evidence=policy_evidence,
            analytics_context=analytics or {},
            recommendation="Escalate for human review using the model evidence and cited policy evidence." if required else "Use the evidence bundle as decision support and apply human review according to analyst workflow.",
            human_review_required=required,
        )

    def run(self, prediction: dict, risk_raising: list[dict], risk_reducing: list[dict], policy_evidence: list[dict], analytics: dict | None = None) -> str:
        package = self.build_package(prediction, risk_raising, risk_reducing, policy_evidence, analytics)
        evidence_text = "\n".join(f"[{x['evidence_id']}] {x['section']}: {x['content']}" for x in policy_evidence)
        raise_text = ", ".join(x.get("Feature", "") for x in risk_raising[:5])
        reduce_text = ", ".join(x.get("Feature", "") for x in risk_reducing[:5])
        analytics_text = str(analytics or "No additional structured analytics requested.")
        if self.llm is None:
            return self._fallback(package, raise_text, reduce_text)
        prompt = f"""You are CredWise AI, an educational banking decision-intelligence assistant.
Use ONLY the supplied evidence. Do not calculate/change the probability, invent facts, invent policy, or approve/reject a loan. Clearly separate model prediction, SHAP evidence, policy evidence, analytics, LLM interpretation, and human decision.

MODEL: {prediction}
SHAP risk-raising: {raise_text}
SHAP risk-reducing: {reduce_text}
POLICY EVIDENCE:\n{evidence_text or 'NONE'}
ANALYTICS:\n{analytics_text}

Write concise headings:
Risk Summary
Model Evidence
Policy Evidence
Customer / Portfolio Context
Decision-Support Recommendation
Human Review

Every policy claim must cite its evidence id. If evidence is absent, say policy evidence is unavailable. The final lending decision belongs to a human reviewer."""
        try:
            return str(self.llm.invoke(prompt).content).strip()
        except Exception:
            return self._fallback(package, raise_text, reduce_text)

    @staticmethod
    def _fallback(package: DecisionPackage, raise_text: str, reduce_text: str) -> str:
        refs = ", ".join(x["evidence_id"] for x in package.policy_evidence) or "No policy evidence retrieved."
        review = "Required" if package.human_review_required else "Recommended according to analyst workflow"
        return (
            f"### Risk Summary\nThe calibrated model estimates a {package.risk_summary['default_probability']:.1%} probability of default and assigns **{package.risk_summary['risk_level']}**.\n\n"
            f"### Model Evidence\nRisk-raising SHAP factors: {raise_text or 'None identified'}. Risk-reducing SHAP factors: {reduce_text or 'None identified'}. SHAP values describe model contribution, not causation.\n\n"
            f"### Policy Evidence\nRetrieved evidence: {refs}. Review the cited source in its original context.\n\n"
            f"### Customer / Portfolio Context\nStructured analytics: {package.analytics_context or 'None supplied.'}\n\n"
            f"### Decision-Support Recommendation\n{package.recommendation}\n\n"
            f"### Human Review\nHuman review: **{review}**. This prototype does not make lending decisions automatically."
        )
