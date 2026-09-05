from __future__ import annotations
import pandas as pd
from models.risk_engine import RiskEngine

class ExplainabilityService:
    def __init__(self, risk_engine: RiskEngine):
        self.risk_engine = risk_engine

    def explain(self, applicant: dict, top_n: int = 5) -> dict:
        df = self.risk_engine.explain(applicant, top_n=top_n)
        raising = df[df["SHAP_Value"] > 0].head(top_n)
        reducing = df[df["SHAP_Value"] < 0].head(top_n)
        return {
            "all_factors": df,
            "risk_raising": raising,
            "risk_reducing": reducing,
            "feature_names": df["Feature"].tolist(),
        }

def clean_feature_name(name: str) -> str:
    return (name.replace("num__", "").replace("cat__", "").replace("_", " ").strip().title())
