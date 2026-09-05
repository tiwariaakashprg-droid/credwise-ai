# CredWise AI — SQL Analytics & Audit Architecture

SQLite is used as both the persistence and small analytical layer.

## Core tables

- `loan_applications`: 16 applicant/loan input fields.
- `risk_predictions`: raw XGBoost score, calibrated probability, risk band, model name.
- `shap_explanations`: feature, feature value, SHAP contribution, impact, direction.
- `risk_reports`: generated decision-support report.
- `human_reviews`: human review outcome and comments.

## Agentic/audit tables

- `policy_evidence`: retrieved policy chunks with source/section/evidence ID and retrieval score.
- `analyst_queries`: natural-language question, generated SQL, execution status, results/errors.
- `agent_audit_log`: trace ID, agent, action, status, input/output summaries, errors.

## Natural-language analytics flow

```text
Analyst question
      ↓
Supervisor
      ↓
SQL Agent
      ↓
SQL generation
      ↓
Read-only validation
      ↓
SQLite execution
      ↓
Structured DataFrame
      ↓
Analyst UI
```

No database result is generated from LLM text. The displayed result is read from SQLite after SQL execution.
