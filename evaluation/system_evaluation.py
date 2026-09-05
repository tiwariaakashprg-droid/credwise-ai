from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationSummary:
    total: int
    passed: int
    failed: int
    pass_rate: float


def summarize(results: list[bool]) -> EvaluationSummary:
    total = len(results)
    passed = sum(bool(x) for x in results)
    return EvaluationSummary(total, passed, total - passed, passed / total if total else 0.0)


def evaluate_completion(records: list[dict[str, Any]]) -> EvaluationSummary:
    """Evaluate only whether an end-to-end run completed without an error.

    This intentionally does not infer quality or correctness from completion alone.
    """
    return summarize([not bool(record.get("error")) for record in records])
