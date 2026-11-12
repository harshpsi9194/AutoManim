"""Vector store backends: ChromaDB adapter and local in-memory cosine store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from manim_rag.models import CodeChunk
from manim_rag.retrieval.embeddings import DeterministicHashEmbedder, EmbeddingProvider


@dataclass
class VectorHit:
    """Single vector search result."""

    chunk_id: str
    score: float
    document: str
    metadata: dict[str, Any]


class VectorStoreBackend(Protocol):
    """Protocol satisfied by both ChromaVectorStore and InMemoryVectorStore."""

    def add_chunks(self, chunks: Sequence[CodeChunk]) -> int:
        """Embed and upsert CodeChunks into the vector store."""
        ...

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Search by vector similarity."""
        ...

    def count(self) -> int:
        """Number of indexed chunks."""
        ...


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for key, expected in where.items():
        if metadata.get(key) != expected:
            return False
    return True


class InMemoryVectorStore:
    """Pure NumPy cosine-similarity vector store for zero-download local runs and tests."""

    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self.embedder: EmbeddingProvider = embedder or DeterministicHashEmbedder()
        self._items: dict[str, tuple[np.ndarray, str, dict[str, Any]]] = {}

    def count(self) -> int:
        return len(self._items)

    def add_chunks(self, chunks: Sequence[CodeChunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        docs = [c.to_document_text() for c in chunk_list]
        embeddings = self.embedder.embed_documents(docs)
        for chunk, doc, emb in zip(chunk_list, docs, embeddings):
            arr = np.asarray(emb, dtype=np.float64)
            norm = float(np.linalg.norm(arr))
            if norm > 0.0:
                arr = arr / norm
            self._items[chunk.chunk_id] = (arr, doc, chunk.to_flat_metadata())
        return len(chunk_list)

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        if not self._items or top_k <= 0:
            return []
        q_vec = np.asarray(self.embedder.embed_query(query_text), dtype=np.float64)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0.0:
            q_vec = q_vec / q_norm

        hits: list[VectorHit] = []
        for chunk_id, (vec, doc, meta) in self._items.items():
            if not _matches_where(meta, where):
                continue
            sim = float(np.dot(q_vec, vec)) if q_norm > 0.0 else 0.0
            hits.append(VectorHit(chunk_id=chunk_id, score=sim, document=doc, metadata=meta))

        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]


class ChromaVectorStore:
    """ChromaDB vector store adapter supporting persistent, ephemeral, or injected clients."""

    def __init__(
        self,
        *,
        collection_name: str = "manimce_knowledge_base",
        persist_directory: str | None = None,
        embedder: EmbeddingProvider | None = None,
        client: Any | None = None,
        collection: Any | None = None,
    ) -> None:
        self.embedder: EmbeddingProvider = embedder or DeterministicHashEmbedder()
        self.collection_name = collection_name

        if collection is not None:
            self._client = client
            self._collection = collection
            return

        if client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "chromadb is not installed. Install with `pip install chromadb` "
                    "or use InMemoryVectorStore for local runs."
                ) from exc
            client = (
                chromadb.PersistentClient(path=persist_directory)
                if persist_directory
                else chromadb.EphemeralClient()
            )

        self._client = client
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())

    def add_chunks(self, chunks: Sequence[CodeChunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        ids = [c.chunk_id for c in chunk_list]
        docs = [c.to_document_text() for c in chunk_list]
        metadatas = [c.to_flat_metadata() for c in chunk_list]
        embeddings = self.embedder.embed_documents(docs)
        self._collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunk_list)

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        total = self.count()
        if total == 0 or top_k <= 0:
            return []

        q_emb = self.embedder.embed_query(query_text)
        kwargs: dict[str, Any] = {
            "query_embeddings": [q_emb],
            "n_results": min(top_k, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        hits: list[VectorHit] = []
        for idx, chunk_id in enumerate(ids):
            dist = float(dists[idx]) if idx < len(dists) else 1.0
            score = 1.0 / (1.0 + max(0.0, dist))
            doc = docs[idx] if idx < len(docs) and docs[idx] is not None else ""
            meta = metas[idx] if idx < len(metas) and metas[idx] is not None else {}
            hits.append(VectorHit(chunk_id=chunk_id, score=score, document=doc, metadata=meta))
        return hits
