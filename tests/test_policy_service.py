from pathlib import Path

from rag.policy_service import PolicyRAGService


def test_section_inference():
    assert PolicyRAGService._infer_section("SECTION: Debt-to-Income\nDTI guidance") == "Debt-to-Income"


def test_policy_service_builds_bm25_when_available():
    service = PolicyRAGService(vector_store=None, policy_path=Path("rag/lending_policy.txt"))
    assert service.policy_path.exists()
