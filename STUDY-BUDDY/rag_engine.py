# rag_engine.py
"""
RAG engine: local embeddings (sentence-transformers) + Pinecone (v5) serverless vector DB.
Provides:
- upsert_documents(docs: list[{"id": str, "text": str}])
- query(query_text: str, top_k: int = 4) -> list[{"id", "score", "text"}]
"""
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_MODEL
)

# Initialize embedding model once
_embedding = SentenceTransformer(EMBEDDING_MODEL)
_dim = _embedding.get_sentence_embedding_dimension()

# Init Pinecone (v5)
_pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
def _ensure_index():
    existing = [i.name for i in _pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        _pc.create_index(
            name=PINECONE_INDEX,
            dimension=_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

_ensure_index()
_index = _pc.Index(PINECONE_INDEX)

def _embed(texts: List[str]) -> List[List[float]]:
    return _embedding.encode(texts, convert_to_numpy=True).tolist()

def upsert_documents(docs: List[Dict[str, str]]) -> None:
    """
    Upsert documents into Pinecone.
    Each doc: {"id": str, "text": str}
    """
    if not docs:
        return
    vectors = _embed([d["text"] for d in docs])
    payload = []
    for d, v in zip(docs, vectors):
        payload.append({
            "id": d["id"],
            "values": v,
            "metadata": {"text": d["text"]}
        })
    _index.upsert(vectors=payload)

def query(query_text: str, top_k: int = 4) -> List[Dict[str, str]]:
    """
    Query Pinecone with an input string. Returns list of dicts with id, score, text.
    """
    qv = _embed([query_text])[0]
    res = _index.query(vector=qv, top_k=top_k, include_metadata=True)
    hits = []
    for m in res.matches or []:
        hits.append({
            "id": m.id,
            "score": float(m.score),
            "text": (m.metadata or {}).get("text", "")
        })
    return hits
