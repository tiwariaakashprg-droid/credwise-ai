from __future__ import annotations
import argparse, json
from pathlib import Path
from rag.policy_service import PolicyRAGService


def evaluate(service: PolicyRAGService, cases: list[dict], k: int = 3):
    rows = []
    for case in cases:
        evidence = service.retrieve(case["query"], top_k=k)
        expected = {x.lower() for x in case.get("expected_sections", [])}
        retrieved = {x.section.lower() for x in evidence}
        hit = bool(expected & retrieved) if expected else None
        rows.append({"query": case["query"], "expected_sections": sorted(expected), "retrieved_sections": sorted(retrieved), "top_k_hit": hit, "evidence_count": len(evidence), "methods": sorted({x.retrieval_method for x in evidence})})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate policy retrieval against a user-supplied labeled query set.")
    parser.add_argument("--cases", required=True, help="JSON file: [{query, expected_sections: [...]}]")
    args = parser.parse_args()
    from app import load_policy_service
    service = load_policy_service()
    if service is None:
        raise RuntimeError("Policy service unavailable.")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    rows = evaluate(service, cases)
    print(json.dumps(rows, indent=2))
    labeled = [r for r in rows if r["top_k_hit"] is not None]
    if labeled:
        print(f"\nTop-K labeled hit rate: {sum(r['top_k_hit'] for r in labeled)}/{len(labeled)}")
