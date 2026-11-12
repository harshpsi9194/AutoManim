"""Hybrid search retriever combining Vector (ChromaDB/In-Memory) and BM25 search."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from manim_rag.chunker.knowledge_base import ManimKnowledgeBase
from manim_rag.config import RetrievalConfig
from manim_rag.models import CodeChunk, RetrievedChunk
from manim_rag.retrieval.bm25 import BM25Index
from manim_rag.retrieval.embeddings import DeterministicHashEmbedder
from manim_rag.retrieval.vector_store import InMemoryVectorStore, VectorStoreBackend


def _normalize_scores(score_map: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1], handling empty or uniform score maps."""
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return {k: 1.0 if v > 0.0 else 0.0 for k, v in score_map.items()}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


class HybridRetriever:
    """Combines vector similarity search and BM25 keyword search via RRF or weighted fusion."""

    def __init__(
        self,
        knowledge_base: ManimKnowledgeBase,
        *,
        vector_store: VectorStoreBackend | None = None,
        bm25_index: BM25Index | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.kb = knowledge_base
        self.config = config or RetrievalConfig()
        self.vector_store: VectorStoreBackend = vector_store or InMemoryVectorStore(
            embedder=DeterministicHashEmbedder(dimension=self.config.embedding_dim)
        )
        self.bm25_index: BM25Index = bm25_index or BM25Index()

        if len(self.kb) > 0:
            self.rebuild_index()

    def rebuild_index(self) -> None:
        """Index all chunks currently in the knowledge base into vector + BM25 stores."""
        chunks = self.kb.chunks
        self.bm25_index.index_chunks(chunks)
        if chunks:
            self.vector_store.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[CodeChunk]) -> int:
        """Add new chunks to the knowledge base and refresh retrieval indices."""
        added = self.kb.add_chunks(chunks)
        self.rebuild_index()
        return added

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid vector + BM25 search and return ranked RetrievedChunks."""
        k = self.config.top_k if top_k is None else top_k
        if len(self.kb) == 0 or not query.strip() or k <= 0:
            return []

        vec_hits = self.vector_store.query(
            query,
            top_k=max(k, self.config.vector_candidates),
            where=where,
        )
        bm25_hits = self.bm25_index.search(
            query,
            top_k=max(k, self.config.bm25_candidates),
            where=where,
        )

        vec_scores: dict[str, float] = {hit.chunk_id: hit.score for hit in vec_hits}
        vec_ranks: dict[str, int] = {hit.chunk_id: idx + 1 for idx, hit in enumerate(vec_hits)}
        bm25_scores: dict[str, float] = {chunk.chunk_id: score for chunk, score in bm25_hits}
        bm25_ranks: dict[str, int] = {
            chunk.chunk_id: idx + 1 for idx, (chunk, _) in enumerate(bm25_hits)
        }

        candidate_ids = set(vec_scores) | set(bm25_scores)
        if not candidate_ids:
            return []

        if self.config.fusion_method == "rrf":
            fused_scores = self._fuse_rrf(candidate_ids, vec_ranks, bm25_ranks)
        else:
            fused_scores = self._fuse_weighted(candidate_ids, vec_scores, bm25_scores)

        results: list[RetrievedChunk] = []
        for chunk_id, fused_score in fused_scores.items():
            chunk = self.kb.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=fused_score,
                    vector_score=vec_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                    vector_rank=vec_ranks.get(chunk_id),
                    bm25_rank=bm25_ranks.get(chunk_id),
                )
            )

        results.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        return results[:k]

    def _fuse_rrf(
        self,
        candidate_ids: set[str],
        vec_ranks: dict[str, int],
        bm25_ranks: dict[str, int],
    ) -> dict[str, float]:
        rrf_k = self.config.rrf_k
        w_vec = self.config.vector_weight
        w_bm25 = self.config.bm25_weight
        scores: dict[str, float] = {}
        for cid in candidate_ids:
            score = 0.0
            if cid in vec_ranks:
                score += w_vec * (1.0 / (rrf_k + vec_ranks[cid]))
            if cid in bm25_ranks:
                score += w_bm25 * (1.0 / (rrf_k + bm25_ranks[cid]))
            scores[cid] = score
        return scores

    def _fuse_weighted(
        self,
        candidate_ids: set[str],
        vec_scores: dict[str, float],
        bm25_scores: dict[str, float],
    ) -> dict[str, float]:
        norm_vec = _normalize_scores(vec_scores)
        norm_bm25 = _normalize_scores(bm25_scores)
        w_vec = self.config.vector_weight
        w_bm25 = self.config.bm25_weight
        denom = w_vec + w_bm25
        if denom <= 0.0:
            w_vec, w_bm25, denom = 0.5, 0.5, 1.0
        scores: dict[str, float] = {}
        for cid in candidate_ids:
            v = norm_vec.get(cid, 0.0)
            b = norm_bm25.get(cid, 0.0)
            scores[cid] = (w_vec * v + w_bm25 * b) / denom
        return scores

    def as_langchain_retriever(self, *, top_k: int | None = None) -> ManimLangChainRetriever:
        """Wrap this HybridRetriever as a LangChain BaseRetriever for LCEL chains."""
        return ManimLangChainRetriever(
            hybrid_retriever=self,
            top_k=self.config.top_k if top_k is None else top_k,
        )


class ManimLangChainRetriever(BaseRetriever):
    """LangChain BaseRetriever adapter around HybridRetriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_retriever: HybridRetriever
    top_k: int = 6

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        hits = self.hybrid_retriever.retrieve(query, top_k=self.top_k)
        return [
            Document(
                page_content=hit.chunk.to_document_text(),
                metadata={
                    **hit.chunk.to_flat_metadata(),
                    "score": hit.score,
                    "vector_score": hit.vector_score,
                    "bm25_score": hit.bm25_score,
                },
            )
            for hit in hits
        ]
