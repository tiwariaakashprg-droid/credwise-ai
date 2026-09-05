from __future__ import annotations
from pathlib import Path
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_policy_documents(path: str | Path) -> list[Document]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n\s*SECTION:\s*", text, flags=re.I)
    docs = []
    for index, raw in enumerate(sections):
        raw = raw.strip()
        if not raw:
            continue
        if index == 0 and raw.startswith("CredWise AI"):
            continue
        lines = raw.splitlines()
        section = lines[0].strip() if lines else "Unknown"
        content = "\n".join(lines[1:]).strip()
        docs.append(Document(page_content=content, metadata={"source": path.name, "section": section, "chunk_id": f"section_{index}"}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    for i, doc in enumerate(chunks, start=1):
        doc.metadata["chunk_id"] = f"chunk_{i}"
    return chunks
