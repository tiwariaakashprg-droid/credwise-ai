from __future__ import annotations
import argparse, json
from pathlib import Path
from analytics.sql_agent import SQLAgent


def evaluate(agent: SQLAgent, cases: list[dict]):
    rows = []
    for case in cases:
        result = agent.run(case["query"])
        expected_status = case.get("expected_status", "success")
        rows.append({"query": case["query"], "status": result.status, "expected_status": expected_status, "sql": result.sql, "passed": result.status == expected_status})
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    args = parser.parse_args()
    rows = evaluate(SQLAgent(), json.loads(Path(args.cases).read_text(encoding="utf-8")))
    print(json.dumps(rows, indent=2))
    print(f"\nExecution success: {sum(r['status']=='success' for r in rows)}/{len(rows)}")
