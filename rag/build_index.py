"""Rebuild the policy FAISS index with provenance metadata.

Run this only when the policy source changes or when rebuilding the index in a
clean environment. The existing index is intentionally preserved in the repo.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import SETTINGS
from rag.ingestion import load_policy_documents


def build_index(output_dir: str | Path = SETTINGS.faiss_path):
    docs = load_policy_documents(SETTINGS.policy_path)
    embeddings = HuggingFaceEmbeddings(model_name=SETTINGS.embedding_model)
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(output_dir))
    return len(docs)

if __name__ == "__main__":
    print(f"Indexed {build_index()} policy chunks into {SETTINGS.faiss_path}")
