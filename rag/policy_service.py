from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from config.settings import SETTINGS

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # optional; FAISS-only remains available
    BM25Okapi = None


@dataclass
class PolicyEvidence:
    evidence_id: str
    source: str
    section: str
    content: str
    score: float | None = None
    chunk_id: str | None = None
    retrieval_method: str = "faiss"

    def to_dict(self):
        return asdict(self)


class PolicyRAGService:
    """Evidence-first policy retrieval with optional FAISS+BM25 hybrid ranking."""

    def __init__(self, vector_store=None, policy_path: Path = SETTINGS.policy_path):
        self.vector_store = vector_store
        self.policy_path = policy_path
        self._bm25 = None
        self._bm25_docs = []
        self._build_bm25_fallback()

    def _build_bm25_fallback(self):
        if BM25Okapi is None or not self.policy_path.exists():
            return
        text = self.policy_path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        self._bm25_docs = chunks
        if chunks:
            self._bm25 = BM25Okapi([re.findall(r"\w+", c.lower()) for c in chunks])

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _bm25_retrieve(self, query: str, top_k: int) -> list[PolicyEvidence]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokens(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        evidence = []
        for idx, score in ranked:
            content = self._bm25_docs[idx]
            section = self._infer_section(content)
            evidence.append(PolicyEvidence(f"policy-bm25-{idx+1}", self.policy_path.name, section, content, float(score), f"bm25_{idx+1}", "bm25"))
        return evidence

    def retrieve(self, query: str, top_k: int = SETTINGS.policy_top_k) -> list[PolicyEvidence]:
        faiss_evidence: list[PolicyEvidence] = []
        if self.vector_store is not None:
            docs_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
            for idx, pair in enumerate(docs_scores, start=1):
                doc, score = pair
                metadata = doc.metadata or {}
                source = metadata.get("source", self.policy_path.name)
                section = metadata.get("section") or self._infer_section(doc.page_content)
                chunk_id = metadata.get("chunk_id", f"faiss_{idx}")
                evidence_id = f"policy-{chunk_id}"
                faiss_evidence.append(PolicyEvidence(evidence_id, source, section, doc.page_content, float(score), chunk_id, "faiss"))

        bm25_evidence = self._bm25_retrieve(query, top_k)
        if not faiss_evidence:
            return bm25_evidence
        if not bm25_evidence:
            return faiss_evidence

        # Reciprocal-rank fusion keeps scores comparable across FAISS distance and BM25 scales.
        merged: dict[str, PolicyEvidence] = {}
        ranks: dict[str, float] = {}
        for rank, item in enumerate(faiss_evidence, 1):
            key = re.sub(r"\s+", " ", item.content.strip().lower())
            merged[key] = item
            ranks[key] = ranks.get(key, 0.0) + 1.0 / (60 + rank)
        for rank, item in enumerate(bm25_evidence, 1):
            key = re.sub(r"\s+", " ", item.content.strip().lower())
            if key not in merged:
                merged[key] = item
            ranks[key] = ranks.get(key, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(merged.items(), key=lambda kv: ranks[kv[0]], reverse=True)[:top_k]
        result = []
        for i, (key, item) in enumerate(ordered, 1):
            item.evidence_id = f"policy-hybrid-{i}"
            item.retrieval_method = "hybrid"
            item.score = ranks[key]
            result.append(item)
        return result

    def context(self, evidence: list[PolicyEvidence]) -> str:
        return "\n\n".join(f"[{item.evidence_id}] Source: {item.source} | Section: {item.section} | Method: {item.retrieval_method}\n{item.content}" for item in evidence)

    @staticmethod
    def _infer_section(text: str) -> str:
        match = re.search(r"SECTION:\s*(.+)", text, flags=re.I)
        return match.group(1).strip() if match else "Unspecified policy section"
