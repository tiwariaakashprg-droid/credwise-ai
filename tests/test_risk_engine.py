import pytest
from models.risk_engine import RiskEngine

APPLICANT = {
    "loan_amnt": 10000.0, "term": 36, "installment": 300.0, "emp_length": 5,
    "home_ownership": "RENT", "annual_inc": 60000.0, "verification_status": "Verified",
    "purpose": "debt_consolidation", "dti": 15.0, "delinq_2yrs": 0,
    "inq_last_6mths": 1, "open_acc": 10, "pub_rec": 0, "revol_bal": 10000.0,
    "revol_util": 40.0, "total_acc": 20,
}

def test_prediction_is_model_generated():
    engine = RiskEngine()
    result = engine.predict(APPLICANT)
    assert 0 <= result.raw_risk_score <= 1
    assert 0 <= result.default_probability <= 1
    assert result.model_name == "XGBoost"
    assert result.risk_level in {"Low Risk", "Moderate Risk", "High Risk"}

def test_missing_feature_rejected():
    engine = RiskEngine()
    bad = APPLICANT.copy(); bad.pop("dti")
    with pytest.raises(ValueError):
        engine.predict(bad)

def test_shap_has_feature_values():
    engine = RiskEngine()
    result = engine.explain(APPLICANT)
    assert not result.empty
    assert "Feature_Value" in result.columns
    assert "Direction" in result.columns
