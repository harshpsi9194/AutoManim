"""Shared pytest fixtures for manim-rag tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from manim_rag.chunker import ManimASTChunker, ManimKnowledgeBase
from manim_rag.retrieval import HybridRetriever, InMemoryVectorStore

SAMPLE_SOURCE_DIR = Path(__file__).resolve().parent.parent / "examples" / "sample_manim_source"


@pytest.fixture
def sample_kb() -> ManimKnowledgeBase:
    """Build a ManimKnowledgeBase from the bundled sample ManimCE source files."""
    kb, result = ManimKnowledgeBase.from_directory(
        SAMPLE_SOURCE_DIR,
        chunker=ManimASTChunker(),
    )
    assert result.errors == []
    assert len(kb) > 0
    return kb


@pytest.fixture
def sample_retriever(sample_kb: ManimKnowledgeBase) -> HybridRetriever:
    """Build an in-memory HybridRetriever over the sample knowledge base."""
    return HybridRetriever(sample_kb, vector_store=InMemoryVectorStore())
