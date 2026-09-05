from explainability.shap_service import ExplainabilityService

class ExplainabilityAgent:
    name = "explainability_agent"
    def __init__(self, service: ExplainabilityService):
        self.service = service

    def run(self, applicant: dict) -> dict:
        result = self.service.explain(applicant)
        return {
            "risk_raising": result["risk_raising"].to_dict(orient="records"),
            "risk_reducing": result["risk_reducing"].to_dict(orient="records"),
            "all_factors": result["all_factors"].to_dict(orient="records"),
        }
