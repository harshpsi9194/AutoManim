"""Tests for BM25, vector stores (InMemory + mocked ChromaDB), and HybridRetriever."""

from __future__ import annotations

from typing import Any

from manim_rag.chunker import ManimKnowledgeBase
from manim_rag.config import RetrievalConfig
from manim_rag.retrieval import (
    BM25Index,
    ChromaVectorStore,
    DeterministicHashEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
    tokenize_code,
)
from manim_rag.retrieval.hybrid import _normalize_scores


class _FakeChromaCollection:
    """Minimal in-memory mock matching the chromadb Collection upsert/query/count API."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[list[float], str, dict[str, Any]]] = {}

    def count(self) -> int:
        return len(self.records)

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self.records[cid] = (emb, doc, meta)

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        q = np.asarray(query_embeddings[0], dtype=np.float64)
        scored: list[tuple[str, float, str, dict[str, Any]]] = []
        for cid, (emb, doc, meta) in self.records.items():
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            v = np.asarray(emb, dtype=np.float64)
            cosine_dist = max(0.0, 1.0 - float(np.dot(q, v)))
            scored.append((cid, cosine_dist, doc, meta))
        scored.sort(key=lambda item: item[1])
        top = scored[:n_results]
        return {
            "ids": [[x[0] for x in top]],
            "distances": [[x[1] for x in top]],
            "documents": [[x[2] for x in top]],
            "metadatas": [[x[3] for x in top]],
        }


def test_tokenize_code_splits_camel_and_snake_case() -> None:
    tokens = tokenize_code("ShowCreation set_color ThreeDScene")
    assert "showcreation" in tokens
    assert "show" in tokens
    assert "creation" in tokens
    assert "set_color" in tokens
    assert "set" in tokens
    assert "color" in tokens
    assert "threedscene" in tokens


def test_bm25_index_finds_exact_symbol_and_filters(sample_kb: ManimKnowledgeBase) -> None:
    bm25 = BM25Index(sample_kb.chunks)
    hits = bm25.search("Axes plot function curve", top_k=3)
    assert hits
    assert any(chunk.qualified_name == "Axes.plot" for chunk, _ in hits)

    filtered = bm25.search("Create", top_k=5, where={"chunk_type": "class"})
    assert filtered
    assert all(chunk.chunk_type.value == "class" for chunk, _ in filtered)
    assert filtered[0][0].name == "Create"


def test_deterministic_hash_embedder_and_in_memory_store(sample_kb: ManimKnowledgeBase) -> None:
    embedder = DeterministicHashEmbedder(dimension=128)
    v1 = embedder.embed_query("Create Circle animation")
    v2 = embedder.embed_query("Create Circle animation")
    assert len(v1) == 128
    assert v1 == v2

    store = InMemoryVectorStore(embedder=embedder)
    store.add_chunks(sample_kb.chunks)
    assert store.count() == len(sample_kb)

    hits = store.query("Create VMobject animation", top_k=3)
    assert hits
    assert any("Create" in h.chunk_id for h in hits)


def test_chroma_vector_store_with_mocked_collection(sample_kb: ManimKnowledgeBase) -> None:
    fake_collection = _FakeChromaCollection()
    chroma_store = ChromaVectorStore(collection=fake_collection)
    retriever = HybridRetriever(sample_kb, vector_store=chroma_store)

    assert chroma_store.count() == len(sample_kb)
    hits = retriever.retrieve("FadeIn mobjects", top_k=3)
    assert hits
    assert any(h.chunk.name == "FadeIn" for h in hits)


def test_hybrid_retriever_rrf_and_weighted_fusion_and_langchain(
    sample_kb: ManimKnowledgeBase,
) -> None:
    rrf_retriever = HybridRetriever(
        sample_kb,
        config=RetrievalConfig(top_k=4, fusion_method="rrf"),
    )
    rrf_hits = rrf_retriever.retrieve("Circle radius surround")
    assert 1 <= len(rrf_hits) <= 4
    assert any("Circle" in h.chunk.qualified_name for h in rrf_hits)

    weighted_retriever = HybridRetriever(
        sample_kb,
        config=RetrievalConfig(
            top_k=4,
            fusion_method="weighted",
            vector_weight=0.4,
            bm25_weight=0.6,
        ),
    )
    w_hits = weighted_retriever.retrieve("Axes plot curve")
    assert w_hits
    assert all(0.0 <= h.score <= 1.0 for h in w_hits)

    lc_retriever = rrf_retriever.as_langchain_retriever(top_k=2)
    docs = lc_retriever.invoke("Transform mobject")
    assert len(docs) == 2
    assert any("Transform" in d.page_content for d in docs)


def test_normalize_scores_edge_cases() -> None:
    assert _normalize_scores({}) == {}
    assert _normalize_scores({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}
    assert _normalize_scores({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    normed = _normalize_scores({"a": 2.0, "b": 6.0})
    assert normed == {"a": 0.0, "b": 1.0}


def test_hybrid_retriever_top_k_zero_and_embedding_dim(sample_kb: ManimKnowledgeBase) -> None:
    retriever = HybridRetriever(
        sample_kb,
        config=RetrievalConfig(top_k=5, embedding_dim=64),
    )
    assert retriever.vector_store.embedder.dimension == 64  # type: ignore[attr-defined]
    assert retriever.retrieve("Circle", top_k=0) == []
    assert retriever.retrieve("Circle", top_k=-2) == []
