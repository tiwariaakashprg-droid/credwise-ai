# CredWise AI — Agentic Banking Decision Intelligence & Explainable Risk Platform

CredWise AI is an **educational/research credit-risk decision-support prototype** that combines machine learning, explainable AI, retrieval-augmented generation, controlled agentic orchestration, SQL analytics, and human-in-the-loop review.

The 2026 extension evolves an existing ML + SHAP + policy-RAG application into a controlled **agentic banking decision-intelligence platform** while preserving the original trained model artifacts and human-review workflow.

> **Important:** CredWise AI is not a production lending system. It does not automatically approve or reject loans. AI-generated outputs are decision support only, and a human reviewer remains the final decision authority.

---

## 1. Problem Statement

Credit-risk models can produce a probability of default, but analysts also need to understand **why** a model produced that result and how the result relates to relevant policy and structured application data.

A useful banking decision-support system therefore needs to combine:

* Model predictions
* Model explanations
* Policy evidence
* Structured database analytics
* Natural-language analyst interaction
* Human review
* Auditable execution history

CredWise AI connects these evidence sources while maintaining clear boundaries between them.

The LLM is **not** the source of truth for model probabilities, SHAP explanations, policy facts, or database results.

---

## 2. Motivation

The project demonstrates how the following technologies can work together in a financial-services decision-support workflow:

* Machine Learning
* Explainable AI
* Retrieval-Augmented Generation
* Agentic orchestration
* Natural-language SQL analytics
* Large Language Models
* Human-in-the-loop review
* Audit logging

The primary design goal is **controlled integration rather than unconstrained LLM decision-making**.

---

## 3. Core Features

### Machine Learning

* Existing XGBoost credit-default model
* Existing probability calibration
* Historical Logistic Regression, Balanced Logistic Regression, Random Forest, XGBoost, and PyTorch benchmark experiments retained in the notebook
* Applicant-level risk prediction
* Configurable risk display bands

### Explainable AI

* SHAP-based model explanations
* Applicant-level explanations
* Global SHAP importance
* Feature values
* SHAP contribution values
* Risk-raising and risk-reducing directions

### Policy RAG

* Existing `all-MiniLM-L6-v2` embeddings
* FAISS semantic retrieval
* Optional BM25 lexical retrieval
* Reciprocal-rank fusion for hybrid retrieval
* Policy evidence provenance
* Evidence IDs
* Source and section metadata
* Chunk identifiers
* Retrieval scores

### Agentic Architecture

* Supervisor / Router
* Risk Agent
* Explainability Agent
* Policy Agent
* SQL Agent
* Decision Intelligence Agent
* LangGraph orchestration when available
* Deterministic fallback routing when LangGraph is unavailable

### SQL Analytics

* Natural-language analyst interface
* Deterministic SQL templates for common queries
* Optional LLM-assisted SQL generation
* SQL validation
* Read-only SQLite execution
* Table allowlisting
* Destructive SQL blocking
* Single-statement enforcement
* Query logging

### Decision Intelligence

* Evidence-based decision package
* Model prediction summary
* SHAP evidence
* Policy evidence
* Structured analytics
* Analyst-friendly narrative reporting
* Local Llama 3.2 through Ollama
* Safe fallback reporting when the LLM is unavailable

### Human-in-the-Loop

* Human reviewer workflow
* Reviewer decision recording
* Reviewer comments
* AI output explicitly treated as decision support

### Auditability

The system records:

* Application inputs
* Risk predictions
* SHAP explanations
* Policy evidence
* Generated reports
* Human reviews
* Analyst questions
* SQL queries
* Agent routing events
* Agent execution status
* Trace IDs

### Engineering

* Configuration through environment variables
* Data-quality warnings
* Automated tests
* Separate evaluation utilities
* Architecture documentation
* Runbook
* Changelog

---

## 4. Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the detailed system design.

```text
                         Analyst
                            |
                            v
                  Supervisor / Router
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
      SQL Agent        Risk Agent         Policy Agent
          |                 |                  |
          v                 v                  v
   SQLite Analytics     XGBoost          MiniLM + FAISS
                        Calibration       + optional BM25
                            |
                            v
                           SHAP
                            |
          +-----------------+------------------+
                            |
                            v
                 Decision Intelligence
                            |
                            v
                       Llama 3.2
                            |
                            v
                     Human Review
                            |
                            v
                      Audit Trail
```

The architecture intentionally separates **evidence generation** from **narrative interpretation**.

---

## 5. Agent Architecture

### 5.1 Supervisor / Router

The Supervisor determines which workflow should handle an analyst request.

Supported routes include:

* Risk analysis
* SQL analytics
* Policy retrieval
* Combined decision intelligence

When LangGraph is installed, the workflow can use a LangGraph state graph.

A deterministic routing fallback is also available for environments where LangGraph is unavailable.

The router records:

* Selected route
* Routing confidence
* Routing rationale
* Trace ID
* Execution status

---

### 5.2 Risk Agent

The Risk Agent wraps the deterministic ML pipeline.

```text
Applicant Features
        |
        v
XGBoost Model
        |
        v
Raw Risk Score
        |
        v
Probability Calibration
        |
        v
Calibrated Default Probability
        |
        v
SHAP Explanation
```

The LLM does not calculate or override the model probability.

The Risk Agent returns structured risk information including:

* Raw model score
* Calibrated probability
* Display risk level
* Model information
* Data-quality warnings
* Risk-raising factors
* Risk-reducing factors
* SHAP evidence

---

### 5.3 Explainability Agent

The Explainability Agent provides a controlled interface around the SHAP service.

For an applicant, the system can record:

* Feature name
* Feature value
* SHAP contribution
* Absolute impact
* Direction of contribution

SHAP represents **model contribution**, not causation.

---

### 5.4 SQL Agent

The SQL Agent allows analysts to ask structured questions using natural language.

Examples:

```text
How many high-risk applicants are present?

What is the average predicted default probability?

Which applicants have the highest predicted default probability?

Show the distribution of predicted risk.

How does risk vary by loan amount or income?
```

The SQL workflow follows:

```text
Natural Language Question
          |
          v
Deterministic Template / LLM SQL Generation
          |
          v
SQL Validation
          |
          v
Read-Only SQLite
          |
          v
Structured Results
```

The application enforces several controls:

* Only `SELECT` / `WITH` queries are allowed
* Destructive SQL keywords are blocked
* Only approved tables can be accessed
* Multiple statements are rejected
* Queries are executed against a read-only SQLite connection
* Analyst queries are logged

Database results come from SQLite rather than being invented by the LLM.

---

### 5.5 Policy Agent

The Policy Agent retrieves relevant lending-policy evidence.

The retrieval pipeline can use:

```text
Policy Knowledge Base
        |
        +--------------------+
        |                    |
        v                    v
      FAISS                 BM25
   Semantic Search       Lexical Search
        |                    |
        +---------+----------+
                  |
                  v
       Reciprocal-Rank Fusion
                  |
                  v
          Policy Evidence
```

Retrieved evidence contains provenance such as:

```text
Evidence ID
Source
Section
Chunk ID
Retrieval Method
Retrieval Score
Content
```

This allows a policy statement presented to the analyst to be traced back to retrieved policy evidence.

---

### 5.6 Decision Intelligence Agent

The Decision Intelligence layer combines evidence from:

* Risk prediction
* SHAP explanations
* Policy retrieval
* Structured SQL analytics

It first constructs a structured evidence package.

Only then is the package passed to the local Llama 3.2 model for narrative generation.

The LLM is therefore used for:

* Interpretation
* Summarization
* Narrative generation
* Analyst-facing explanation

It is **not** used as the source of truth for:

* Risk probability
* SHAP values
* Policy facts
* SQL results
* Final lending decisions

A high-risk display band can flag a case for human review, but it is **not a lending approval/decline rule**.

---

## 6. Machine Learning

The deployed project uses the existing XGBoost artifact with probability calibration.

```text
Applicant Features
        |
        v
Existing XGBoost Pipeline
        |
        v
Raw Risk Score
        |
        v
Existing Probability Calibrator
        |
        v
Calibrated Default Probability
```

The project notebook also contains historical model experiments and comparisons.

These include:

* Logistic Regression
* Balanced Logistic Regression
* Random Forest
* XGBoost
* PyTorch-based experiments

Those experiments are retained for reference and are not automatically retrained or replaced by the agentic extension.

The deployed risk engine uses the preserved XGBoost model artifact and calibration artifact.

---

## 7. Explainability

SHAP is the primary explainability mechanism.

For each applicant assessment, CredWise can record:

```text
Feature
Feature Value
SHAP Contribution
Absolute Impact
Direction
```

Example interpretation:

```text
High DTI
    |
    +--> Positive SHAP contribution
         |
         +--> Increases model-predicted risk
```

and:

```text
Strong income / favorable feature
    |
    +--> Negative SHAP contribution
         |
         +--> Reduces model-predicted risk
```

SHAP values should be interpreted as contributions to the model output, **not as causal effects**.

---

## 8. Policy Retrieval-Augmented Generation

CredWise reuses the existing lending-policy retrieval system based on:

* `all-MiniLM-L6-v2`
* FAISS

The system can additionally use:

* BM25 lexical retrieval
* Reciprocal-rank fusion

The policy knowledge base is project-specific and contains sections covering topics such as:

* Debt-to-income ratio
* Credit utilization
* Credit inquiries
* Delinquency history
* Income and repayment capacity
* Loan amount and installment
* Employment history
* Loan term
* Decision policy

Retrieved policy evidence is stored with provenance metadata.

---

## 9. SQL Analytics

SQLite serves two purposes:

1. Application persistence
2. Structured analytical querying

Important database tables include:

```text
loan_applications
risk_predictions
shap_explanations
risk_reports
human_reviews
policy_evidence
analyst_queries
agent_audit_log
```

The SQL Agent can answer questions such as:

```text
How many high-risk applicants are present?

What is the average predicted default probability?

Which applicants have the highest predicted default probability?

Show the distribution of predicted risk.

How does risk vary by loan amount?

How does risk vary by income?
```

The SQL layer is intentionally read-only at the application level.

---

## 10. Decision Intelligence

The decision layer creates a structured evidence package before generating a narrative.

The package separates:

```text
Risk Summary
Model Evidence
SHAP Evidence
Policy Evidence
Analytics Context
Recommendation / Review Signal
Human Review Status
Disclaimer
```

This prevents the LLM from silently replacing deterministic system outputs.

The resulting narrative is intended to help an analyst or reviewer understand the available evidence.

It is not an automated underwriting decision.

---

## 11. Human-in-the-Loop

CredWise explicitly maintains human oversight.

The workflow is:

```text
AI Analysis
    |
    v
Evidence Package
    |
    v
Narrative Report
    |
    v
Human Reviewer
    |
    v
Final Decision
```

The system can record:

* Reviewer decision
* Reviewer comments
* Review timestamp
* Associated application/report

The AI output remains decision support.

A human reviewer remains the final decision authority.

---

## 12. Auditability

CredWise maintains a traceable chain across the decision-support workflow.

```text
Application Input
       |
       v
Model Prediction
       |
       v
SHAP Explanation
       |
       v
Policy Evidence
       |
       v
SQL / Structured Context
       |
       v
Decision Intelligence Report
       |
       v
Human Review
```

The database stores relevant events and evidence so that an analyst can inspect how an output was produced.

Agent execution events include trace information such as:

* Trace ID
* Agent / route
* Execution status
* Routing rationale
* Execution metadata

---

## 13. Data Quality

CredWise performs non-fatal applicant data-quality checks.

Examples include:

* Negative numerical values
* Invalid DTI values
* Invalid utilization values
* Zero income
* Other suspicious input conditions

Warnings are surfaced to the analyst without silently modifying the applicant's data.

The goal is to make questionable inputs visible before interpreting model output.

---

## 14. Evaluation

Evaluation is separated by system layer.

| Layer     | Evaluation                                                                         |
| --------- | ---------------------------------------------------------------------------------- |
| ML        | ROC-AUC, PR-AUC, precision, recall, F1, Brier score, calibration, confusion matrix |
| RAG       | Labeled retrieval relevance and Top-K hit rate                                     |
| SQL Agent | SQL validation and execution success                                               |
| Agent     | Routing accuracy, confidence, rationale, workflow success                          |
| System    | Workflow completion, failure rate, and latency where measured                      |

Evaluation utilities are available in:

```text
evaluation/
```

### ML Evaluation

`evaluation/ml_evaluation.py`

Evaluates a supplied labeled dataset and reports metrics such as:

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Raw Brier score
* Calibrated Brier score
* Confusion matrix

The script does not fabricate benchmark results.

---

### RAG Evaluation

`evaluation/rag_evaluation.py`

Uses labeled evaluation cases to measure retrieval performance, including Top-K hit rate.

---

### SQL Evaluation

`evaluation/sql_evaluation.py`

Evaluates SQL validation and execution behavior using predefined test cases.

---

### Agent Evaluation

`evaluation/agent_evaluation.py`

Evaluates routing behavior using predefined analyst requests.

---

### System Evaluation

`evaluation/system_evaluation.py`

Provides workflow-level reliability information such as completion and failure summaries.

System completion should not be interpreted as evidence of model quality.

---

## 15. No Fabricated Results

CredWise intentionally avoids claiming new benchmark numbers without running the appropriate evaluation dataset.

The repository contains:

* Evaluation utilities
* Labeled evaluation cases
* Historical notebook experiments

The original project README/notebook contains historical model-comparison results and an earlier small RAG sanity-check experiment.

Those results should be treated as **historical project results**, not as production-level performance guarantees.

New ML, RAG, SQL, or agent benchmark claims should be generated by running the corresponding evaluation utilities on an appropriate dataset.

---

## 16. Technology Stack

| Technology            | Role                             |
| --------------------- | -------------------------------- |
| Python                | Core implementation              |
| XGBoost               | Credit-risk prediction           |
| scikit-learn          | Preprocessing and calibration    |
| SHAP                  | Model explainability             |
| PyTorch               | Historical benchmark experiments |
| Sentence Transformers | Policy embeddings                |
| FAISS                 | Semantic retrieval               |
| BM25                  | Lexical retrieval                |
| LangChain             | LLM/retrieval integration        |
| LangGraph             | Agent workflow orchestration     |
| Ollama                | Local LLM runtime                |
| Llama 3.2             | Narrative generation             |
| SQLite                | Persistence and analytics        |
| Pandas                | Data processing                  |
| Streamlit             | Analyst interface                |
| Pytest                | Automated testing                |

Each technology has a concrete role in the system.

No additional infrastructure is required solely for resume keywords.

---

## 17. Project Structure

```text
credwise-ai/
│
├── agents/
│   ├── decision_agent.py
│   ├── explainability_agent.py
│   ├── policy_agent.py
│   ├── risk_agent.py
│   └── supervisor.py
│
├── analytics/
│   └── sql_agent.py
│
├── config/
│   └── settings.py
│
├── evaluation/
│   ├── agent_evaluation.py
│   ├── ml_evaluation.py
│   ├── rag_evaluation.py
│   ├── sql_evaluation.py
│   ├── system_evaluation.py
│   └── *_eval_cases.json
│
├── explainability/
│   └── shap_service.py
│
├── models/
│   ├── data_quality.py
│   ├── risk_engine.py
│   ├── xgboost_pipeline.pkl
│   ├── probability_calibrator.pkl
│   └── shap_explainer.pkl
│
├── rag/
│   ├── build_index.py
│   ├── ingestion.py
│   ├── lending_policy.txt
│   └── policy_service.py
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── tests/
│   ├── test_data_quality.py
│   ├── test_database.py
│   ├── test_end_to_end.py
│   ├── test_policy_service.py
│   ├── test_risk_engine.py
│   ├── test_routing.py
│   └── test_sql_agent.py
│
├── notebooks/
│   └── 01_data_preparation.ipynb
│
├── app.py
├── credwise_ai_app.py
├── database.py
├── credwise.db
├── ARCHITECTURE.md
├── SQL_ARCHITECTURE.md
├── RUNBOOK.md
├── CHANGELOG.md
├── VERSION.txt
├── requirements.txt
└── .env.example
```

---

## 18. Installation

### Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 19. Local LLM Setup

Narrative generation uses local Llama 3.2 through Ollama.

Install Ollama separately and pull the model:

```bash
ollama pull llama3.2
```

The application can still provide deterministic/fallback outputs when the LLM is unavailable for supported workflows.

---

## 20. Configuration

Copy the example environment file if required:

```text
.env.example
```

to:

```text
.env
```

The `.env` file is intentionally ignored by Git.

Configuration can include:

* LLM model
* LLM temperature
* Embedding model
* Policy Top-K
* Risk display thresholds

Never commit API keys, credentials, or other secrets.

---

## 21. Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

The original entry point is also preserved for backward compatibility:

```bash
streamlit run credwise_ai_app.py
```

The dashboard provides functionality for:

* Overview
* Applicant Risk
* SHAP explanations
* Policy evidence
* Analyst natural-language queries
* SQL analytics
* Decision Intelligence
* Human review
* Audit history

---

## 22. Testing

Run the automated test suite from the repository root:

```bash
pytest -q
```

The final repository was validated with the included automated test suite in the development environment.

The validated test suite currently contains **13 passing tests**.

Serialized model warnings may appear when the installed scikit-learn/XGBoost versions differ from the versions used to create the original model artifacts.

The artifacts are intentionally preserved rather than silently retrained.

---

## 23. Validation

Recommended validation sequence:

### 1. Run tests

```bash
pytest -q
```

### 2. Validate Python compilation

```bash
python -m compileall .
```

### 3. Start the application

```bash
streamlit run app.py
```

### 4. Test Applicant Risk

Create or inspect an applicant assessment and verify:

* Prediction
* Calibrated probability
* Risk display
* SHAP factors
* Data-quality warnings

### 5. Test Policy Retrieval

Verify that retrieved policy results contain:

* Evidence ID
* Section
* Source
* Retrieval method
* Score
* Content

### 6. Test SQL Analyst

Try:

```text
How many high-risk applicants are present?
```

and:

```text
What is the average predicted default probability?
```

Verify that the returned results come from SQLite.

### 7. Test Supervisor Routing

Examples:

```text
Why is this applicant high risk?
```

Expected route:

```text
Risk
```

```text
How many high-risk applicants are there?
```

Expected route:

```text
SQL
```

```text
What lending policy applies here?
```

Expected route:

```text
Policy
```

```text
Summarize this applicant for a human reviewer.
```

Expected route:

```text
Decision Intelligence
```

### 8. Test Human Review

Verify that reviewer decisions and comments are persisted.

### 9. Test Audit Trail

Verify that agent execution and analyst query events are recorded.

---

## 24. Limitations

CredWise AI has several important limitations.

### Model Artifact Compatibility

The supplied model artifacts were serialized in an older dependency environment.

Compatibility should therefore be checked before relying on the artifacts.

### Synthetic / Project-Specific Policy

The lending policy knowledge base is a synthetic/project-specific knowledge base.

It is not an actual bank's credit policy.

### Risk Thresholds

Risk display bands are configurable application thresholds.

They are not:

* Regulatory thresholds
* Bank-approved underwriting thresholds
* Credit policy rules

### Local Database

The included SQLite database is a small local development/portfolio database.

It is not a production data platform.

### Natural-Language SQL

Natural-language SQL generation for questions outside deterministic templates may depend on the local LLM.

Generated SQL is validated before execution.

### RAG Evaluation

RAG relevance must be evaluated on a meaningful labeled dataset before drawing general conclusions about retrieval quality.

### Fairness and Compliance

The project does not claim:

* Fairness certification
* Regulatory compliance
* Production reliability
* Lending suitability
* Legal compliance
* Real-world underwriting accuracy

---

## 25. Future Work

Potential future extensions include:

* Larger labeled policy-retrieval benchmark
* Formal agent task-success benchmark
* Model monitoring
* Data and concept drift analysis
* Fairness and bias assessment where legally and ethically appropriate
* Policy document versioning
* Policy lifecycle management
* Role-based access control
* Stronger audit security
* Reproducible model retraining and export pipeline
* More comprehensive portfolio analytics
* Automated evaluation dashboards

---

## 26. Final Design Principle

CredWise treats each evidence source as authoritative only within its own domain.

```text
XGBoost + Calibration
        ↓
     Prediction

SHAP
        ↓
Model Contribution

FAISS / BM25
        ↓
Policy Evidence

SQLite
        ↓
Structured Analytics

Llama 3.2
        ↓
Constrained Interpretation

Human Reviewer
        ↓
Final Lending Decision
```

This separation is the core safety, reliability, and auditability principle of the project.

The system is designed so that the LLM **interprets evidence rather than replacing the systems that generate authoritative evidence**.

---

## 27. Disclaimer

CredWise AI is an academic/research project for demonstrating:

* Explainable credit-risk analytics
* Retrieval-augmented generation
* Controlled agentic workflows
* Natural-language SQL analytics
* Human-in-the-loop decision support
* Auditable AI workflows

It is **not**:

* Financial advice
* A credit underwriting service
* A production banking system
* An automated lending decision system

Do not use CredWise AI to make real-world lending decisions.
