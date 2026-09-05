from __future__ import annotations

import argparse
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from config.settings import FEATURES, SETTINGS


def evaluate(data_path: str, target: str = "loan_default", threshold: float | None = None):
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X, y = df[FEATURES], df[target].astype(int)
    pipeline = joblib.load(SETTINGS.model_path)
    calibrator = joblib.load(SETTINGS.calibrator_path)
    raw = pipeline.predict_proba(X)[:, 1]
    calibrated = calibrator.predict(raw)
    cutoff = SETTINGS.moderate_risk_threshold if threshold is None else threshold
    pred = (calibrated >= cutoff).astype(int)
    metrics = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc_raw": roc_auc_score(y, raw),
        "pr_auc_raw": average_precision_score(y, raw),
        "brier_raw": brier_score_loss(y, raw),
        "brier_calibrated": brier_score_loss(y, calibrated),
        "classification_threshold": cutoff,
    }
    return metrics, confusion_matrix(y, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the existing CredWise XGBoost artifact on a held-out labeled dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="loan_default")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    metrics, cm = evaluate(args.data, args.target, args.threshold)
    print(pd.Series(metrics).to_string())
    print("\nConfusion matrix:\n", cm)
