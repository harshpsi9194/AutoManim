"""Core data models shared across chunking, retrieval, generation, and execution."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Kind of Python AST entity extracted from ManimCE source."""

    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    MODULE = "module"


class CodeChunk(BaseModel):
    """Structured representation of an AST-parsed ManimCE symbol."""

    chunk_id: str
    chunk_type: ChunkType
    name: str
    qualified_name: str
    module_path: str
    file_path: str
    parent_class: str | None = None
    signature: str = ""
    decorators: list[str] = Field(default_factory=list)
    bases: list[str] = Field(default_factory=list)
    docstring: str = ""
    source: str = ""
    start_line: int = 1
    end_line: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_document_text(self) -> str:
        """Render chunk into a rich searchable text representation."""
        parts = [f"{self.chunk_type.value}: {self.qualified_name}"]
        if self.module_path:
            parts.append(f"module: {self.module_path}")
        if self.signature:
            parts.append(f"signature: {self.signature}")
        if self.bases:
            parts.append(f"bases: {', '.join(self.bases)}")
        if self.decorators:
            parts.append(f"decorators: {', '.join(self.decorators)}")
        if self.docstring:
            parts.append(f"docstring:\n{self.docstring}")
        if self.source:
            parts.append(f"source:\n{self.source}")
        return "\n".join(parts)

    def to_flat_metadata(self) -> dict[str, str | int | float | bool]:
        """Return ChromaDB-compatible scalar metadata."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "module_path": self.module_path,
            "file_path": self.file_path,
            "parent_class": self.parent_class or "",
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


class RetrievedChunk(BaseModel):
    """A retrieved CodeChunk along with vector, BM25, and fused scores."""

    chunk: CodeChunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None


class FailureCategory(str, Enum):
    """Classified ManimCE execution failure categories."""

    API_ERROR = "api_error"
    ARGUMENT_ERROR = "argument_error"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class FailureDiagnostic(BaseModel):
    """Structured diagnostics extracted from failed ManimCE execution."""

    category: FailureCategory
    error_type: str
    error_message: str
    line_number: int | None = None
    column_offset: int | None = None
    offending_line: str | None = None
    extracted_symbols: list[str] = Field(default_factory=list)
    suggested_query: str = ""
    diagnostic_summary: str = ""
    raw_traceback: str = ""


class ExecutionResult(BaseModel):
    """Result from executing a ManimCE script inside the Docker sandbox."""

    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0
    output_video_path: str | None = None
    command: list[str] = Field(default_factory=list)


class AttemptRecord(BaseModel):
    """Trace record for one generate-execute-diagnose iteration."""

    attempt_number: int
    prompt_used: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_query: str = ""
    generated_code: str = ""
    scene_name: str | None = None
    execution: ExecutionResult | None = None
    diagnostic: FailureDiagnostic | None = None


class PipelineResult(BaseModel):
    """Final output returned by the RAG-based Manim generation pipeline."""

    success: bool
    user_prompt: str
    final_code: str
    scene_name: str | None = None
    output_video_path: str | None = None
    total_attempts: int = 0
    attempts: list[AttemptRecord] = Field(default_factory=list)
    last_diagnostic: FailureDiagnostic | None = None
