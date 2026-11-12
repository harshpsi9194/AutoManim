"""Hybrid search package combining ChromaDB/in-memory vectors and BM25."""

from manim_rag.retrieval.bm25 import BM25Index, tokenize_code
from manim_rag.retrieval.embeddings import (
    DeterministicHashEmbedder,
    EmbeddingProvider,
    OpenAICompatibleEmbedder,
)
from manim_rag.retrieval.hybrid import HybridRetriever, ManimLangChainRetriever
from manim_rag.retrieval.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorHit,
    VectorStoreBackend,
)

__all__ = [
    "BM25Index",
    "ChromaVectorStore",
    "DeterministicHashEmbedder",
    "EmbeddingProvider",
    "HybridRetriever",
    "InMemoryVectorStore",
    "ManimLangChainRetriever",
    "OpenAICompatibleEmbedder",
    "VectorHit",
    "VectorStoreBackend",
    "tokenize_code",
]
