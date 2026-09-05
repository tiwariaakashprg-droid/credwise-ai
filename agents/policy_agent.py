from rag.policy_service import PolicyRAGService

class PolicyAgent:
    name = "policy_agent"
    def __init__(self, service: PolicyRAGService):
        self.service = service

    def run(self, query: str, top_k: int = 3) -> dict:
        evidence = self.service.retrieve(query, top_k=top_k)
        return {"evidence": [item.to_dict() for item in evidence], "context": self.service.context(evidence)}
