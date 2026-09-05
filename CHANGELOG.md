# Changelog

## 2.0.0-final — 2026

Final portfolio-grade extension of the original CredWise AI prototype.

### Added
- Controlled Supervisor routing with route confidence and rationale.
- LangGraph workflow with deterministic fallback.
- Read-only SQL analytics using a dedicated SQLite read-only connection.
- SQL table allowlist, destructive-keyword checks, single-statement enforcement, row limits, and validation notes.
- Hybrid policy retrieval using FAISS plus optional BM25 reciprocal-rank fusion.
- Policy evidence provenance: source, section, chunk ID, retrieval method, and score.
- Structured decision-intelligence package separating model evidence, policy evidence, analytics, recommendation, and human-review status.
- Applicant data-quality warnings without changing model outputs.
- Empirical SHAP aggregation from stored assessments in the dashboard.
- Expanded tests for routing, SQL safety, policy retrieval, data quality, read-only analytics, and existing ML artifacts.
- Stronger evaluation utilities that distinguish completion from quality/correctness.

### Preserved
- Original XGBoost model artifact.
- Original probability calibrator.
- Original SHAP artifact.
- Existing FAISS index.
- Existing SQLite database and human-review flow.
- Existing Streamlit entry point.

### Not claimed
No new performance, retrieval, agent, or business KPI is presented as a measured result unless it is generated from an appropriate evaluation dataset.
