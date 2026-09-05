from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import SETTINGS

DB_PATH = SETTINGS.db_path


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def get_readonly_connection():
    """Open SQLite in read-only mode for analytical agents."""
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _add_column_if_missing(cursor, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS loan_applications (
        application_id INTEGER PRIMARY KEY AUTOINCREMENT,
        loan_amnt REAL NOT NULL, term INTEGER NOT NULL, installment REAL NOT NULL,
        emp_length INTEGER NOT NULL, home_ownership TEXT NOT NULL, annual_inc REAL NOT NULL,
        verification_status TEXT NOT NULL, purpose TEXT NOT NULL, dti REAL NOT NULL,
        delinq_2yrs INTEGER NOT NULL, inq_last_6mths INTEGER NOT NULL, open_acc INTEGER NOT NULL,
        pub_rec INTEGER NOT NULL, revol_bal REAL NOT NULL, revol_util REAL NOT NULL,
        total_acc INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS risk_predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        application_id INTEGER NOT NULL, raw_risk_score REAL NOT NULL,
        default_probability REAL NOT NULL, risk_level TEXT NOT NULL,
        model_name TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (application_id) REFERENCES loan_applications(application_id) ON DELETE CASCADE
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS shap_explanations (
        explanation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL, feature_name TEXT NOT NULL,
        feature_value TEXT, shap_value REAL NOT NULL, impact REAL NOT NULL,
        direction TEXT, FOREIGN KEY (prediction_id) REFERENCES risk_predictions(prediction_id) ON DELETE CASCADE
    )""")
    _add_column_if_missing(cursor, "shap_explanations", "feature_value", "TEXT")
    _add_column_if_missing(cursor, "shap_explanations", "direction", "TEXT")

    cursor.execute("""CREATE TABLE IF NOT EXISTS risk_reports (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_id INTEGER NOT NULL,
        report_text TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES risk_predictions(prediction_id) ON DELETE CASCADE
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS human_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT, application_id INTEGER NOT NULL,
        reviewer_decision TEXT NOT NULL, reviewer_comment TEXT,
        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (application_id) REFERENCES loan_applications(application_id) ON DELETE CASCADE
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS policy_evidence (
        evidence_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER, evidence_id TEXT NOT NULL, source TEXT NOT NULL,
        section TEXT, content TEXT NOT NULL, retrieval_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES risk_predictions(prediction_id) ON DELETE SET NULL
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS analyst_queries (
        query_id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT NOT NULL,
        route TEXT, generated_sql TEXT, execution_status TEXT,
        result_json TEXT, error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS agent_audit_log (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL,
        agent_name TEXT NOT NULL, action TEXT NOT NULL, status TEXT NOT NULL,
        input_summary TEXT, output_summary TEXT, error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_app ON risk_predictions(application_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_shap_prediction ON shap_explanations(prediction_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_trace ON agent_audit_log(trace_id)")
    conn.commit()
    conn.close()


def save_application(applicant: dict) -> int:
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("""INSERT INTO loan_applications (
        loan_amnt, term, installment, emp_length, home_ownership, annual_inc,
        verification_status, purpose, dti, delinq_2yrs, inq_last_6mths,
        open_acc, pub_rec, revol_bal, revol_util, total_acc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", tuple(applicant[k] for k in [
            "loan_amnt","term","installment","emp_length","home_ownership","annual_inc",
            "verification_status","purpose","dti","delinq_2yrs","inq_last_6mths","open_acc",
            "pub_rec","revol_bal","revol_util","total_acc"]))
    application_id = int(cursor.lastrowid); conn.commit(); conn.close(); return application_id


def save_prediction(application_id, raw_risk_score, default_probability, risk_level, model_name="XGBoost") -> int:
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("""INSERT INTO risk_predictions
        (application_id, raw_risk_score, default_probability, risk_level, model_name)
        VALUES (?, ?, ?, ?, ?)""", (application_id, float(raw_risk_score), float(default_probability), risk_level, model_name))
    prediction_id = int(cursor.lastrowid); conn.commit(); conn.close(); return prediction_id


def save_shap_explanations(prediction_id: int, shap_df: pd.DataFrame) -> None:
    conn = get_connection(); cursor = conn.cursor()
    rows = []
    for i in range(len(shap_df)):
        row = shap_df.iloc[i]
        rows.append((prediction_id, str(row["Feature"]), str(row.get("Feature_Value", "")), float(row["SHAP_Value"]), float(row["Impact"]), str(row.get("Direction", ""))))
    cursor.executemany("""INSERT INTO shap_explanations
        (prediction_id, feature_name, feature_value, shap_value, impact, direction)
        VALUES (?, ?, ?, ?, ?, ?)""", rows)
    conn.commit(); conn.close()


def save_report(prediction_id: int, report_text: str) -> int:
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("INSERT INTO risk_reports (prediction_id, report_text) VALUES (?, ?)", (prediction_id, report_text))
    row_id = int(cursor.lastrowid); conn.commit(); conn.close(); return row_id


def save_human_review(application_id: int, decision: str, comment: str | None) -> int:
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("INSERT INTO human_reviews (application_id, reviewer_decision, reviewer_comment) VALUES (?, ?, ?)", (application_id, decision, comment))
    row_id = int(cursor.lastrowid); conn.commit(); conn.close(); return row_id


def save_policy_evidence(prediction_id: int | None, evidence: list[dict[str, Any]]) -> None:
    if not evidence: return
    conn = get_connection(); cursor = conn.cursor()
    cursor.executemany("""INSERT INTO policy_evidence
        (prediction_id, evidence_id, source, section, content, retrieval_score)
        VALUES (?, ?, ?, ?, ?, ?)""", [(prediction_id, x["evidence_id"], x["source"], x.get("section"), x["content"], x.get("score")) for x in evidence])
    conn.commit(); conn.close()


def save_agent_event(trace_id, agent_name, action, status, input_summary=None, output_summary=None, error_message=None):
    conn = get_connection(); cursor = conn.cursor()
    cursor.execute("""INSERT INTO agent_audit_log
        (trace_id, agent_name, action, status, input_summary, output_summary, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)""", (trace_id, agent_name, action, status, input_summary, output_summary, error_message))
    conn.commit(); conn.close()


def save_analyst_query(query_text, route, generated_sql, execution_status, result=None, error_message=None):
    conn = get_connection(); cursor = conn.cursor()
    result_json = json.dumps(result, default=str) if result is not None else None
    cursor.execute("""INSERT INTO analyst_queries
        (query_text, route, generated_sql, execution_status, result_json, error_message)
        VALUES (?, ?, ?, ?, ?, ?)""", (query_text, route, generated_sql, execution_status, result_json, error_message))
    conn.commit(); conn.close()


def get_application_history(limit=20):
    conn = get_connection()
    query = """SELECT a.application_id, a.loan_amnt, a.purpose, a.annual_inc,
        p.default_probability, p.risk_level, p.model_name, a.created_at
        FROM loan_applications a LEFT JOIN risk_predictions p ON a.application_id=p.application_id
        ORDER BY a.application_id DESC LIMIT ?"""
    df = pd.read_sql_query(query, conn, params=(limit,)); conn.close(); return df


def get_dashboard_stats():
    conn = get_connection(); cursor = conn.cursor()
    total = cursor.execute("SELECT COUNT(*) FROM loan_applications").fetchone()[0]
    low = cursor.execute("SELECT COUNT(*) FROM risk_predictions WHERE risk_level='Low Risk'").fetchone()[0]
    moderate = cursor.execute("SELECT COUNT(*) FROM risk_predictions WHERE risk_level='Moderate Risk'").fetchone()[0]
    high = cursor.execute("SELECT COUNT(*) FROM risk_predictions WHERE risk_level='High Risk'").fetchone()[0]
    reviews = cursor.execute("SELECT COUNT(*) FROM human_reviews").fetchone()[0]
    avg = cursor.execute("SELECT AVG(default_probability) FROM risk_predictions").fetchone()[0] or 0.0
    conn.close()
    return {"total_applications": total, "low_risk": low, "moderate_risk": moderate, "high_risk": high, "total_reviews": reviews, "average_probability": avg}


def get_db_schema() -> str:
    conn = get_connection(); rows = conn.execute("""SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name""").fetchall(); conn.close()
    return "\n\n".join(f"TABLE {name}:\n{sql}" for name, sql in rows)


def get_global_shap_importance(limit: int = 15) -> pd.DataFrame:
    """Aggregate stored applicant explanations into an empirical dashboard view."""
    conn = get_connection()
    query = """SELECT feature_name, AVG(impact) AS mean_abs_shap, COUNT(*) AS observations
               FROM shap_explanations GROUP BY feature_name
               ORDER BY mean_abs_shap DESC LIMIT ?"""
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df
