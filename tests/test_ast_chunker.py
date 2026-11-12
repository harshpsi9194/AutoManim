"""Tests for the Python AST-based ManimCE source chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from manim_rag.chunker import ManimASTChunker
from manim_rag.models import ChunkType


SAMPLE_SOURCE = '''"""Sample module docstring."""

from dataclasses import dataclass

@dataclass
class OuterMobject(BaseMobject):
    """Outer mobject doc."""

    radius: float = 1.0

    @property
    def diameter(self) -> float:
        """Compute diameter."""
        return self.radius * 2

    def _private_helper(self) -> None:
        return None

    class InnerConfig:
        """Nested configuration class."""

        def validate(self, *, strict: bool = True) -> bool:
            return strict


async def render_async(scene_name: str, /, quality: str = "l", *flags: str, **opts: int) -> str:
    """Async helper function."""
    def local_inner() -> None:
        pass
    return scene_name
'''


def test_chunk_source_extracts_classes_methods_nested_and_async() -> None:
    chunker = ManimASTChunker()
    chunks = chunker.chunk_source(
        SAMPLE_SOURCE,
        file_path="manim/mobject/outer.py",
        module_path="manim.mobject.outer",
    )

    by_qname = {c.qualified_name: c for c in chunks}
    assert "OuterMobject" in by_qname
    assert "OuterMobject.diameter" in by_qname
    assert "OuterMobject.InnerConfig" in by_qname
    assert "OuterMobject.InnerConfig.validate" in by_qname
    assert "render_async" in by_qname

    # Private methods and local nested functions are excluded by default.
    assert "OuterMobject._private_helper" not in by_qname
    assert "render_async.local_inner" not in by_qname

    outer = by_qname["OuterMobject"]
    assert outer.chunk_type == ChunkType.CLASS
    assert outer.bases == ["BaseMobject"]
    assert outer.decorators == ["dataclass"]
    assert outer.docstring == "Outer mobject doc."
    assert outer.start_line == 5  # starts at @dataclass decorator, not class line
    assert outer.source.startswith("@dataclass")

    prop = by_qname["OuterMobject.diameter"]
    assert prop.chunk_type == ChunkType.METHOD
    assert prop.parent_class == "OuterMobject"
    assert prop.decorators == ["property"]
    assert prop.signature == "def diameter(self) -> float"

    inner_validate = by_qname["OuterMobject.InnerConfig.validate"]
    assert inner_validate.parent_class == "OuterMobject.InnerConfig"
    assert inner_validate.signature == "def validate(self, *, strict: bool = True) -> bool"

    async_fn = by_qname["render_async"]
    assert async_fn.chunk_type == ChunkType.ASYNC_FUNCTION
    assert (
        async_fn.signature
        == "async def render_async(scene_name: str, /, quality: str = 'l', *flags: str, **opts: int) -> str"
    )


def test_chunk_source_includes_private_module_and_nested_when_enabled() -> None:
    chunker = ManimASTChunker(
        include_private=True,
        include_module_chunk=True,
        include_nested_functions=True,
    )
    chunks = chunker.chunk_source(SAMPLE_SOURCE, module_path="manim.sample")
    qnames = {c.qualified_name for c in chunks}

    assert "manim.sample" in qnames
    assert "OuterMobject._private_helper" in qnames
    assert "render_async.local_inner" in qnames


def test_duplicate_definitions_produce_unique_chunk_ids() -> None:
    code = """
if True:
    def create_shape(x: int) -> int:
        return x
else:
    def create_shape(x: str) -> str:
        return x
"""
    chunks = ManimASTChunker().chunk_source(code, module_path="manim.cond")
    assert len(chunks) == 2
    assert len({c.chunk_id for c in chunks}) == 2


def test_syntax_error_skip_and_raise(tmp_path: Path) -> None:
    bad_file = tmp_path / "broken.py"
    bad_file.write_text("def broken(:\n    pass\n", encoding="utf-8")
    good_file = tmp_path / "good.py"
    good_file.write_text("class Circle:\n    pass\n", encoding="utf-8")

    # Skip mode records error and still parses good files.
    res = ManimASTChunker(on_syntax_error="skip").chunk_directory(tmp_path)
    assert res.files_processed == 1
    assert len(res.chunks) == 1
    assert res.chunks[0].name == "Circle"
    assert len(res.errors) == 1
    assert res.errors[0].error_type == "SyntaxError"
    assert res.errors[0].line_number == 1

    # Raise mode propagates SyntaxError.
    with pytest.raises(SyntaxError):
        ManimASTChunker(on_syntax_error="raise").chunk_file(bad_file)


def test_posonly_signatures_with_varargs_and_kwonly() -> None:
    code = """
def f1(a: int, /, *args: str) -> None:
    pass

def f2(a: int, /, *, b: int = 1) -> int:
    return a + b

def f3(a: int, /) -> int:
    return a
"""
    chunks = ManimASTChunker().chunk_source(code, module_path="manim.sigs")
    sigs = {c.name: c.signature for c in chunks}
    assert sigs["f1"] == "def f1(a: int, /, *args: str) -> None"
    assert sigs["f2"] == "def f2(a: int, /, *, b: int = 1) -> int"
    assert sigs["f3"] == "def f3(a: int, /) -> int"


def test_non_utf8_file_is_skipped_cleanly(tmp_path: Path) -> None:
    bad_encoding = tmp_path / "latin1_invalid.py"
    bad_encoding.write_bytes(b"\xff\xfe\x00def broken(): pass")

    res = ManimASTChunker(on_syntax_error="skip").chunk_file(bad_encoding)
    assert res.files_processed == 0
    assert res.chunks == []
    assert len(res.errors) == 1
    assert res.errors[0].error_type == "UnicodeDecodeError"
