# BankRAG – Explainable AI Credit Risk & Loan Intelligence System

BankRAG is an explainable AI-based credit-risk decision-support system that combines machine learning, SHAP explainability, Retrieval-Augmented Generation (RAG), and a local Large Language Model.

## Features

- Credit default risk prediction using XGBoost
- Probability calibration
- Comparison with Logistic Regression, Random Forest, and PyTorch Neural Network
- SHAP-based global and applicant-level explainability
- Banking lending-policy retrieval using FAISS
- MiniLM sentence embeddings
- Grounded credit-risk reports using Llama 3.2
- Interactive Streamlit dashboard
- Human-review-oriented decision support

## Architecture

```text
Applicant Information
        ↓
XGBoost Credit Risk Model
        ↓
Probability Calibration
        ↓
SHAP Explainability
        ↓
MiniLM Embeddings + FAISS
        ↓
Relevant Lending Policy
        ↓
Llama 3.2
        ↓
Grounded Credit Risk Report
        ↓
Human Review
```

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8204 | 0.4635 | 0.0271 | 0.0512 | 0.6784 | 0.3078 |
| Balanced Logistic Regression | 0.6463 | 0.2767 | 0.6056 | 0.3798 | 0.6779 | 0.3064 |
| Random Forest | 0.7256 | 0.3109 | 0.4391 | 0.3640 | 0.6790 | 0.3064 |
| XGBoost | 0.6525 | 0.2829 | 0.6143 | 0.3873 | **0.6944** | 0.3230 |
| PyTorch Neural Network | 0.6419 | 0.2802 | **0.6387** | **0.3895** | 0.6930 | **0.3231** |

XGBoost is used as the production risk model because of its strong overall discrimination performance and suitability for structured tabular credit-risk data.

## RAG Evaluation

A small manual Top-1 retrieval sanity evaluation was performed using five lending-policy queries.

**Relevant retrievals: 5/5**

This is a small relevance check and should not be interpreted as a general RAG accuracy benchmark.

## Technology Stack

- Python
- XGBoost
- PyTorch
- Scikit-learn
- SHAP
- FAISS
- Sentence Transformers (`all-MiniLM-L6-v2`)
- LangChain
- Ollama
- Llama 3.2
- Streamlit

## Installation

```bash
pip install -r requirements.txt
```

Install Ollama separately and pull Llama 3.2:

```bash
ollama pull llama3.2
```

## Run

```bash
streamlit run bankrag_app.py
```

## Disclaimer

BankRAG is an educational and research-oriented decision-support system. Its predictions and generated explanations should not be used as the sole basis for real-world lending decisions. Final decisions require appropriate human review.