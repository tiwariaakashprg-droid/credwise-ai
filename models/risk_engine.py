from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap

from config.settings import SETTINGS, FEATURES
from models.data_quality import validate_applicant_values

@dataclass
class RiskResult:
    raw_risk_score: float
    default_probability: float
    risk_level: str
    model_name: str
    data_quality_warnings: list[str] | None = None

    def to_dict(self):
        return asdict(self)

class RiskEngine:
    """Controlled wrapper around the existing CredWise ML artifacts.

    The LLM never calculates or overrides the probability.
    """
    def __init__(self, model_path: Path = SETTINGS.model_path,
                 calibrator_path: Path = SETTINGS.calibrator_path,
                 shap_path: Path = SETTINGS.shap_path):
        self.pipeline = joblib.load(model_path)
        self.calibrator = joblib.load(calibrator_path)
        self.shap_explainer = joblib.load(shap_path)

    @staticmethod
    def validate_applicant(applicant: Mapping[str, Any]) -> None:
        missing = [name for name in FEATURES if name not in applicant]
        if missing:
            raise ValueError(f"Missing required applicant fields: {', '.join(missing)}")

    @staticmethod
    def classify(probability: float) -> str:
        if probability < SETTINGS.low_risk_threshold:
            return "Low Risk"
        if probability < SETTINGS.moderate_risk_threshold:
            return "Moderate Risk"
        return "High Risk"

    def predict(self, applicant: Mapping[str, Any]) -> RiskResult:
        self.validate_applicant(applicant)
        frame = pd.DataFrame([{name: applicant[name] for name in FEATURES}])
        raw = float(self.pipeline.predict_proba(frame)[0, 1])
        calibrated = float(np.clip(self.calibrator.predict([raw])[0], 0.0, 1.0))
        return RiskResult(raw, calibrated, self.classify(calibrated), "XGBoost", validate_applicant_values(applicant))

    def explain(self, applicant: Mapping[str, Any], top_n: int = 5) -> pd.DataFrame:
        self.validate_applicant(applicant)
        frame = pd.DataFrame([{name: applicant[name] for name in FEATURES}])
        preprocessor = self.pipeline.named_steps["preprocessor"]
        processed = preprocessor.transform(frame)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()
        explanation = self.shap_explainer(processed)
        names = list(preprocessor.get_feature_names_out())
        values = np.asarray(explanation.values[0], dtype=float)
        result = pd.DataFrame({"Feature": names, "SHAP_Value": values})
        result["Impact"] = result["SHAP_Value"].abs()
        result["Direction"] = np.where(result["SHAP_Value"] > 0, "risk_raising", "risk_reducing")
        result["Feature_Value"] = [self._feature_value(name, applicant) for name in names]
        return result.sort_values("Impact", ascending=False).reset_index(drop=True).head(top_n * 4)

    @staticmethod
    def _feature_value(transformed_name: str, applicant: Mapping[str, Any]) -> Any:
        name = transformed_name.replace("num__", "").replace("cat__", "")
        for field in FEATURES:
            if name == field or name.startswith(field + "_"):
                return applicant.get(field)
        return None
