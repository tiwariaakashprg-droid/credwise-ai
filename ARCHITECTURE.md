# CredWise AI — 2026 Architecture

CredWise is an educational/research credit-risk decision-support prototype. It is **not** a production lending system and does not make autonomous lending decisions.

## Controlled source-of-truth architecture

```text
                         Banking Analyst
                              |
                    Natural-language question
                              |
                              v
                    Supervisor / Router
                       /      |       \
                      /       |        \
                     v        v         v
                  SQL Agent Risk Agent Policy Agent
                     |         |          |
                     v         v          v
                  SQLite   XGBoost      FAISS
                  Analytics Calibration  Policy KB
                              |
                              v
                            SHAP
                              |
                     Evidence bundle
                              |
                              v
                   Decision Intelligence
                              |
                              v
                         Llama 3.2
                              |
                              v
                    Analyst-friendly report
                              |
                              v
                         Human Review
                              |
                              v
                         Audit Trail
```

## Why these agents?

- **Supervisor/Router:** selects the appropriate workflow.
- **SQL Agent:** translates supported analyst questions into validated read-only SQL and executes it against the real SQLite database.
- **Risk Agent:** wraps the existing XGBoost + calibrated probability + SHAP pipeline. It does not delegate probability calculation to an LLM.
- **Policy Agent:** wraps the existing FAISS policy retriever and adds evidence provenance.
- **Decision Intelligence Agent:** combines structured evidence into a reviewer-oriented report. It cannot override the model probability.
- **Explainability service:** is intentionally a deterministic service used by the Risk Agent rather than an unconstrained LLM agent.

## Evidence boundaries

| Evidence | Source of truth |
|---|---|
| Default probability | Existing calibrated XGBoost pipeline |
| Model drivers | Existing SHAP explainer |
| Policy statements | Retrieved policy chunks |
| Portfolio/application analytics | SQLite queries |
| Narrative interpretation | Llama 3.2, constrained by supplied evidence |
| Final lending decision | Human reviewer |

## Safety controls

1. SQL Agent allows only a single `SELECT`/`WITH` statement.
2. SQL destructive keywords and unauthorized tables are rejected.
3. RAG evidence retains source, section, chunk ID, and retrieval score where available.
4. The report prompt explicitly forbids invented facts and automatic approval/rejection.
5. Agent events and analyst queries are persisted in SQLite.
6. Human review is preserved as the final decision authority.
7. Risk thresholds are configurable **display bands only**, not claimed banking policy.

## Existing components preserved

The existing XGBoost artifact, probability calibrator, SHAP artifact, MiniLM/FAISS index, Llama/Ollama integration, SQLite database, Streamlit workflow, and human-review flow are reused rather than rebuilt.
