from dataclasses import dataclass
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    db_path: Path = ROOT_DIR / "credwise.db"
    model_path: Path = ROOT_DIR / "models" / "xgboost_pipeline.pkl"
    calibrator_path: Path = ROOT_DIR / "models" / "probability_calibrator.pkl"
    shap_path: Path = ROOT_DIR / "models" / "shap_explainer.pkl"
    faiss_path: Path = ROOT_DIR / "faiss_index"
    policy_path: Path = ROOT_DIR / "rag" / "lending_policy.txt"
    embedding_model: str = os.getenv("CREDWISE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = os.getenv("CREDWISE_LLM_MODEL", "llama3.2")
    llm_temperature: float = float(os.getenv("CREDWISE_LLM_TEMPERATURE", "0"))
    policy_top_k: int = int(os.getenv("CREDWISE_POLICY_TOP_K", "3"))
    low_risk_threshold: float = float(os.getenv("CREDWISE_LOW_RISK_THRESHOLD", "0.15"))
    moderate_risk_threshold: float = float(os.getenv("CREDWISE_MODERATE_RISK_THRESHOLD", "0.30"))

SETTINGS = Settings()

FEATURES = [
    "loan_amnt", "term", "installment", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
]

RISK_TABLES = {
    "loan_applications", "risk_predictions", "shap_explanations", "risk_reports",
    "human_reviews", "agent_audit_log", "policy_evidence", "analyst_queries",
}
