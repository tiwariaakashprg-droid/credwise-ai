from __future__ import annotations
import argparse, json
from pathlib import Path
from agents.supervisor import SupervisorAgent


def evaluate(supervisor: SupervisorAgent, cases: list[dict]):
    rows = []
    for case in cases:
        predicted, confidence, reason = supervisor.route_details(case["query"])
        expected = case["expected_route"]
        rows.append({"query": case["query"], "predicted_route": predicted, "expected_route": expected, "confidence": confidence, "reason": reason, "passed": predicted == expected})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()
    rows = evaluate(SupervisorAgent(), json.loads(Path(args.cases).read_text(encoding="utf-8")))
    print(json.dumps(rows, indent=2))
    print(f"\nRouting accuracy: {sum(r['passed'] for r in rows)}/{len(rows)}")
