# Evaluation

The evaluation layer intentionally separates five concerns:

1. **ML:** classification, ranking, calibration, and confusion matrix.
2. **RAG:** labeled retrieval hit rate at configurable Top-K.
3. **SQL Agent:** execution success and expected-status checks.
4. **Agent routing:** deterministic route accuracy against a labeled case set.
5. **System:** latency and failure-rate measurement for a supplied callable.

The repository does not include a new labeled held-out dataset, so no new ML benchmark number is claimed here. The historical model-comparison values remain in the original notebook/README as prior project results; rerun `ml_evaluation.py` on a held-out dataset to produce a current benchmark.
