# CredWise AI — Runbook

## 1. Environment

Use Python 3.11 (recommended) and install:

```bash
pip install -r requirements.txt
```

The supplied model artifacts were serialized with an older scikit-learn/XGBoost environment. If loading raises compatibility errors, recreate the original training environment or retrain/export the artifacts rather than silently changing their behavior.

## 2. Local LLM

Install Ollama separately and make sure the selected model is available locally:

```bash
ollama pull llama3.2
```

The application remains partially usable without the LLM, but narrative generation and arbitrary NL-to-SQL generation require the configured LLM.

## 3. Start

```bash
streamlit run app.py
```

The legacy command also remains supported:

```bash
streamlit run credwise_ai_app.py
```

## 4. Tests

```bash
pytest -q
```

## 5. Evaluation

ML evaluation requires a labeled held-out CSV containing the 16 model features and a binary target:

```bash
python evaluation/ml_evaluation.py --data path/to/held_out.csv --target loan_default
```

Policy retrieval evaluation requires a labeled JSON query set. A starter set is provided at `evaluation/rag_eval_cases.json`.

SQL evaluation:

```bash
python evaluation/sql_evaluation.py --cases evaluation/sql_eval_cases.json
```

Routing evaluation:

```bash
python evaluation/agent_evaluation.py --cases evaluation/agent_eval_cases.json
```

No evaluation script creates a fake benchmark; metrics are calculated only from supplied data/cases.
