from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from config.settings import SETTINGS, RISK_TABLES
from database import get_db_schema, get_readonly_connection, save_analyst_query

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REPLACE|VACUUM|PRAGMA|REINDEX|TRUNCATE|GRANT|REVOKE)\b",
    re.I,
)
TABLE_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.I)


@dataclass
class SQLResult:
    query: str
    sql: str | None
    dataframe: pd.DataFrame
    status: str
    error: str | None = None
    validation_notes: list[str] | None = None


class SQLAgent:
    """Natural-language to read-only SQLite analytics with validation and guardrails."""

    name = "sql_agent"

    def __init__(self, llm=None):
        self.llm = llm

    def _heuristic_sql(self, query: str) -> str | None:
        q = query.lower()
        if "how many" in q and ("high-risk" in q or "high risk" in q):
            return "SELECT COUNT(*) AS high_risk_applicants FROM risk_predictions WHERE risk_level = 'High Risk'"
        if "average" in q and ("default probability" in q or "risk probability" in q):
            return "SELECT AVG(default_probability) AS average_default_probability FROM risk_predictions"
        if "highest" in q and ("default probability" in q or "risk" in q):
            return ("SELECT a.application_id, p.default_probability, p.risk_level, a.loan_amnt, "
                    "a.annual_inc, a.purpose FROM risk_predictions p "
                    "JOIN loan_applications a ON a.application_id=p.application_id "
                    "ORDER BY p.default_probability DESC LIMIT 10")
        if "distribution" in q and "risk" in q:
            return "SELECT risk_level, COUNT(*) AS applicant_count FROM risk_predictions GROUP BY risk_level ORDER BY applicant_count DESC"
        if "loan amount" in q and "risk" in q:
            return ("SELECT a.loan_amnt, AVG(p.default_probability) AS average_default_probability, "
                    "COUNT(*) AS applications FROM loan_applications a "
                    "JOIN risk_predictions p ON p.application_id=a.application_id "
                    "GROUP BY a.loan_amnt ORDER BY a.loan_amnt")
        if "income" in q and "risk" in q:
            return ("SELECT a.annual_inc, AVG(p.default_probability) AS average_default_probability, "
                    "COUNT(*) AS applications FROM loan_applications a "
                    "JOIN risk_predictions p ON p.application_id=a.application_id "
                    "GROUP BY a.annual_inc ORDER BY a.annual_inc")
        return None

    def generate_sql(self, query: str) -> str:
        heuristic = self._heuristic_sql(query)
        if heuristic:
            return heuristic
        if self.llm is None:
            raise RuntimeError("No LLM is available for this analytical question. Try a supported analytics question or configure Ollama.")
        prompt = f"""You generate read-only SQLite analytics SQL for CredWise AI.

Database schema:
{get_db_schema()}

User question:
{query}

Rules:
- Return exactly one SQL statement.
- SELECT or WITH only.
- Never modify data.
- Use only tables and columns present in the schema.
- No markdown fences.
- Prefer concise analytical queries.
- Include LIMIT 100 unless the result is a single aggregate.
- Never invent database results.
SQL:"""
        text = str(self.llm.invoke(prompt).content).strip()
        text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.I).strip()
        return text

    def validate_sql(self, sql: str) -> list[str]:
        statement = sql.strip().rstrip(";").strip()
        if not statement:
            raise ValueError("Generated SQL is empty.")
        if ";" in statement:
            raise ValueError("Multiple SQL statements are not allowed.")
        if not re.match(r"^(SELECT|WITH)\b", statement, re.I):
            raise ValueError("Only SELECT/WITH analytical queries are allowed.")
        if FORBIDDEN.search(statement):
            raise ValueError("Potentially destructive SQL operation detected.")
        tables = {name.lower() for name in TABLE_PATTERN.findall(statement)}
        unknown = tables - {t.lower() for t in RISK_TABLES}
        if unknown:
            raise ValueError(f"Query references unauthorized tables: {', '.join(sorted(unknown))}")
        if re.search(r"\bLIMIT\s+([0-9]{4,}|[1-9][0-9]{3,})\b", statement, re.I):
            raise ValueError("Result limit is too large.")
        if not re.search(r"\bLIMIT\s+\d+\b", statement, re.I) and not re.search(r"\b(COUNT|AVG|SUM|MIN|MAX)\s*\(", statement, re.I):
            statement += " LIMIT 100"
        return ["SELECT/WITH only", "allowlisted tables", "read-only SQLite connection", "maximum 500 returned rows"]

    def execute(self, query: str, sql: str) -> SQLResult:
        try:
            notes = self.validate_sql(sql)
            conn = get_readonly_connection()
            try:
                df = pd.read_sql_query(sql.rstrip(";"), conn)
            finally:
                conn.close()
            if len(df) > 500:
                raise ValueError("Query returned more than 500 rows; add a narrower filter or LIMIT.")
            result = SQLResult(query, sql.rstrip(";"), df, "success", validation_notes=notes)
            save_analyst_query(query, "sql", result.sql, "success", df.to_dict(orient="records"))
            return result
        except Exception as exc:
            result = SQLResult(query, sql, pd.DataFrame(), "error", str(exc))
            save_analyst_query(query, "sql", sql, "error", error_message=str(exc))
            return result

    def run(self, query: str) -> SQLResult:
        try:
            sql = self.generate_sql(query)
        except Exception as exc:
            save_analyst_query(query, "sql", None, "error", error_message=str(exc))
            return SQLResult(query, None, pd.DataFrame(), "error", str(exc))
        return self.execute(query, sql)
