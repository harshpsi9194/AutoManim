"""Tests for ManimKnowledgeBase lookup, filtering, and JSON persistence."""

from __future__ import annotations

from pathlib import Path

from manim_rag.chunker import ManimKnowledgeBase
from manim_rag.models import ChunkType


def test_knowledge_base_lookup_filter_and_roundtrip(
    sample_kb: ManimKnowledgeBase, tmp_path: Path
) -> None:
    circles = sample_kb.find_by_symbol("Circle")
    assert any(c.qualified_name == "Circle" for c in circles)

    circle_methods = sample_kb.filter(chunk_type=ChunkType.METHOD, parent_class="Circle")
    assert {m.name for m in circle_methods} == {"__init__", "surround"}

    anim_chunks = sample_kb.filter(module_prefix="animations")
    assert any(c.name == "Create" for c in anim_chunks)

    save_path = tmp_path / "kb" / "manim_kb.json"
    sample_kb.save(save_path)
    loaded = ManimKnowledgeBase.load(save_path)
    assert len(loaded) == len(sample_kb)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in sample_kb.chunks]
