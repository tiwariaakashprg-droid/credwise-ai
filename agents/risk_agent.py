from models.risk_engine import RiskEngine
from explainability.shap_service import ExplainabilityService

class RiskAgent:
    name = "risk_agent"
    def __init__(self, risk_engine: RiskEngine):
        self.risk_engine = risk_engine
        self.explainability = ExplainabilityService(risk_engine)

    def run(self, applicant: dict) -> dict:
        prediction = self.risk_engine.predict(applicant)
        explanation = self.explainability.explain(applicant)
        return {
            "prediction": prediction.to_dict(),
            "risk_raising": explanation["risk_raising"].to_dict(orient="records"),
            "risk_reducing": explanation["risk_reducing"].to_dict(orient="records"),
            "all_shap": explanation["all_factors"].to_dict(orient="records"),
        }
