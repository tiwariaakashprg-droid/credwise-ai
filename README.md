# CredWise AI — Agentic Banking Decision Intelligence & Explainable Risk Platform

CredWise AI is an **educational/research credit-risk decision-support prototype** evolved from an existing ML + SHAP + policy-RAG application. The 2026 extension adds controlled agentic orchestration, read-only natural-language SQL analytics, policy evidence provenance, decision-intelligence reporting, audit logging, and a stronger analyst interface while preserving the original model artifacts and human-review workflow.

> **Important:** CredWise AI is not a production lending system. It does not automatically approve or reject loans. A human reviewer remains the final decision authority.

## 1. Problem Statement

Credit-risk models can produce a probability without making the underlying evidence easy for an analyst to inspect. Banking analysts also need to combine model output with policy evidence and structured portfolio/application context. CredWise connects these evidence sources without allowing the LLM to become the source of truth for model probabilities, policy facts, or database results.

## 2. Motivation

The project demonstrates how ML, explainable AI, retrieval-augmented generation, agentic orchestration, SQL analytics, and human-in-the-loop review can work together in a financial-services decision-support workflow.

## 3. Core Features

- Existing XGBoost credit-default model.
- Existing probability calibration.
- Existing Logistic Regression, Balanced Logistic Regression, Random Forest, XGBoost, and PyTorch benchmark work retained in the notebook.
- Applicant-level and global SHAP explainability.
- MiniLM + FAISS lending-policy retrieval, extended with optional BM25 hybrid fusion.
- Policy evidence IDs, source/section/chunk metadata, retrieval method, and retrieval scores.
- Local Llama 3.2 via Ollama for grounded narrative reporting.
- SQLite persistence and analytical queries.
- Validated read-only SQL Agent.
- Risk Agent wrapping the deterministic ML/SHAP pipeline.
- Policy Agent wrapping the deterministic retrieval pipeline.
- Supervisor/Router implemented with LangGraph when installed, with a safe deterministic fallback for minimal environments.
- Decision Intelligence reporting layer.
- Human-in-the-loop review.
- Agent audit trail with trace IDs, routing rationale, and execution status.
- Applicant data-quality warnings and configurable risk display bands.
- Analyst natural-language interface.
- Pytest coverage for core services.
- Separate evaluation utilities for ML, RAG, SQL, routing, and system reliability.

## 4. Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the detailed design.

```text
Analyst
   ↓
Supervisor / Router
   ├── SQL Agent → SQLite Analytics
   ├── Risk Agent → XGBoost → Calibration → SHAP
   └── Policy Agent → MiniLM + FAISS/BM25 → Policy Evidence
                         ↓
                Decision Intelligence
                         ↓
                     Llama 3.2
                         ↓
                   Human Review
                         ↓
                    Audit Trail
```

## 5. Agent Architecture

### Supervisor / Router

Routes an analyst request to SQL analytics, policy retrieval, applicant risk analysis, or the combined decision-intelligence workflow.

### Risk Agent

Wraps the existing XGBoost pipeline, calibrator, and SHAP explainer. The LLM never calculates or overrides the default probability.

### SQL Agent

Generates or selects SQL for analyst questions, validates it as a single read-only `SELECT`/`WITH` query, restricts table access, executes it against the actual SQLite database, and returns structured results.

### Policy Agent

Retrieves relevant policy chunks using the existing FAISS index and an optional BM25 lexical retriever. Reciprocal-rank fusion combines the retrieval rankings when both are available. Results carry source, section, chunk, retrieval-method, and score metadata.

### Decision Intelligence Agent

Combines model, SHAP, policy, and structured analytics into an analyst-friendly narrative. It is explicitly prohibited from making the lending decision.

## 6. Machine Learning

The production artifact remains:

```text
Applicant features
      ↓
Existing XGBoost pipeline
      ↓
Raw risk score
      ↓
Existing probability calibrator
      ↓
Calibrated default probability
```

The existing project notebook contains prior validation/test experiments and model comparisons. Those results are not regenerated or altered merely for the agentic extension.

## 7. Explainability

SHAP is the primary explanation mechanism.

For an applicant, CredWise records:

- feature name
- feature value
- SHAP contribution
- absolute impact
- risk-raising/risk-reducing direction

The UI displays the strongest positive and negative contributors. SHAP is interpreted as model contribution, not causation.

## 8. Policy RAG

The existing `all-MiniLM-L6-v2` + FAISS pipeline is reused. Retrieved evidence now carries provenance fields such as:

```text
Evidence ID
Source
Section
Chunk ID
Retrieval score
Content
```

A policy statement should be traceable to retrieved evidence.

## 9. SQL Analytics

SQLite has been upgraded from a persistence-only role into a small analytical layer.

Example analyst questions:

- How many high-risk applicants are present?
- What is the average predicted default probability?
- Which applicants have the highest predicted default probability?
- Show the distribution of predicted risk.
- How does risk vary by loan amount or income?

SQL execution is read-only at the application layer and results are taken from the database rather than invented by the LLM.

## 10. Decision Intelligence

The decision layer produces a structured evidence package before narrative generation. It separates model prediction, SHAP drivers, policy evidence, structured analytics, recommendation, and human-review status. A high-risk display band can flag a case for human review, but it is not a banking approval/decline rule.

## 11. Human-in-the-Loop

The system records reviewer decisions and comments. The AI output is always described as decision support. Human review remains the final decision authority.

## 12. Auditability

The database records:

- application inputs
- predictions
- SHAP explanations
- policy evidence
- reports
- human reviews
- analyst questions and SQL
- agent trace events

This creates a traceable chain from input to model output, evidence, interpretation, and review.

## 13. Evaluation

Evaluation is deliberately separated into:

| Layer | What is measured |
|---|---|
| ML | ROC-AUC, PR-AUC, precision, recall, F1, Brier/calibration, confusion matrix |
| RAG | labeled retrieval relevance / Top-K hit rate |
| SQL Agent | SQL validation and execution success |
| Agent | routing accuracy and workflow success |
| System | latency, failure rate, and human-review usefulness |

### No fabricated results

The repository does **not** claim new ML/RAG/agent benchmark numbers without running an appropriate evaluation set. Use the scripts in `evaluation/` with held-out/labeled data to produce current results.

The original project README/notebook contains historical model-comparison and five-query RAG sanity-check results. Those should be treated as historical project results, not as evidence of production-level performance.

## 14. Technology Stack

- Python
- XGBoost
- scikit-learn
- SHAP
- PyTorch
- Sentence Transformers
- FAISS
- LangChain
- LangGraph
- Ollama / Llama 3.2
- SQLite
- Pandas
- Streamlit
- Pytest

Each technology has a concrete role; no additional infrastructure is required solely for resume keywords.

## 15. Project Structure

```text
credwise-ai/
├── agents/
├── analytics/
├── config/
├── database/
├── explainability/
├── evaluation/
├── models/
├── rag/
├── tests/
├── notebooks/
├── app.py
├── credwise_ai_app.py
├── database.py
├── ARCHITECTURE.md
├── SQL_ARCHITECTURE.md
└── RUNBOOK.md
```

## 16. Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Configure local Ollama if narrative generation is desired:

```bash
ollama pull llama3.2
```

## 16. Usage

```bash
streamlit run app.py
```

The original entry point is preserved:

```bash
streamlit run credwise_ai_app.py
```

Run tests:

```bash
pytest -q
```

## 17. Limitations

- The supplied model artifacts are educational artifacts and were serialized in an older dependency environment; compatibility should be checked before relying on them.
- The policy knowledge base is a synthetic/project-specific knowledge base, not a bank's actual credit policy.
- Risk bands are configurable display thresholds, not regulatory or bank-approved underwriting thresholds.
- The included database is a small local SQLite store, not a production data platform.
- Natural-language SQL generation depends on the local LLM for questions outside the deterministic templates.
- RAG relevance must be evaluated on a meaningful labeled set before drawing general conclusions.
- No claim of fairness, regulatory compliance, production reliability, or lending suitability is made.

## 18. Future Work

- Larger labeled retrieval benchmark.
- Formal agent task-success benchmark.
- Model monitoring and drift analysis.
- Fairness/bias assessment where legally and ethically appropriate.
- Better policy document lifecycle/versioning.
- Role-based access control and stronger audit security.
- Reproducible model retraining/export pipeline.

## 19. Disclaimer

CredWise AI is an academic/research project for demonstrating explainable credit-risk analytics, retrieval-augmented generation, and controlled agentic workflows. It is **not** financial advice, a credit underwriting service, or production banking software. Do not use it to make real lending decisions.


## 17. Validation

Run the automated tests from the repository root:

```bash
pytest -q
```

The final repository was validated with the included test suite. Serialized model warnings may appear when the environment's scikit-learn/XGBoost versions differ from the versions used to create the original artifacts; the artifacts are intentionally preserved rather than silently retrained.

## 18. Final Design Principle

CredWise treats each evidence source as authoritative only for its own domain:

```text
XGBoost + calibration → prediction
SHAP                 → model contribution
FAISS/BM25 policy    → policy evidence
SQLite               → structured analytics
Llama 3.2            → constrained interpretation
Human reviewer       → final lending decision
```

This separation is the core safety and auditability principle of the project.
