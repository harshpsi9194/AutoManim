"""ManimCE knowledge base management and JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from manim_rag.chunker.ast_chunker import ChunkingResult, ManimASTChunker
from manim_rag.models import ChunkType, CodeChunk


class ManimKnowledgeBase:
    """In-memory and JSON-serializable store of AST-parsed ManimCE CodeChunks."""

    def __init__(self, chunks: Iterable[CodeChunk] | None = None) -> None:
        self._chunks: dict[str, CodeChunk] = {}
        if chunks is not None:
            self.add_chunks(chunks)

    @property
    def chunks(self) -> list[CodeChunk]:
        """Return all stored chunks in insertion order."""
        return list(self._chunks.values())

    def __len__(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: Iterable[CodeChunk]) -> int:
        """Insert or update chunks in the knowledge base."""
        added = 0
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            added += 1
        return added

    def get(self, chunk_id: str) -> CodeChunk | None:
        """Look up a chunk by its unique ID."""
        return self._chunks.get(chunk_id)

    def find_by_symbol(self, symbol_name: str) -> list[CodeChunk]:
        """Find chunks whose short name or qualified name matches symbol_name."""
        return [
            chunk
            for chunk in self._chunks.values()
            if chunk.name == symbol_name or chunk.qualified_name == symbol_name
        ]

    def filter(
        self,
        *,
        chunk_type: ChunkType | None = None,
        parent_class: str | None = None,
        module_prefix: str | None = None,
    ) -> list[CodeChunk]:
        """Filter stored chunks by AST metadata attributes."""
        results: list[CodeChunk] = []
        for chunk in self._chunks.values():
            if chunk_type is not None and chunk.chunk_type != chunk_type:
                continue
            if parent_class is not None and chunk.parent_class != parent_class:
                continue
            if module_prefix is not None and not chunk.module_path.startswith(module_prefix):
                continue
            results.append(chunk)
        return results

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        chunker: ManimASTChunker | None = None,
        exclude_dirs: tuple[str, ...] = ("__pycache__", ".git", ".venv", "tests"),
    ) -> tuple[ManimKnowledgeBase, ChunkingResult]:
        """Build a knowledge base directly by AST-chunking a ManimCE directory."""
        active_chunker = chunker or ManimASTChunker()
        result = active_chunker.chunk_directory(directory, exclude_dirs=exclude_dirs)
        return cls(result.chunks), result

    def save(self, path: str | Path) -> Path:
        """Persist knowledge base chunks to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "count": len(self._chunks),
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks.values()],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ManimKnowledgeBase:
        """Load a previously saved knowledge base from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = [CodeChunk.model_validate(item) for item in data.get("chunks", [])]
        return cls(chunks)
