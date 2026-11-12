"""RAG-Based ManimCE Code Generator."""

from manim_rag.chunker import ChunkingResult, ManimASTChunker, ManimKnowledgeBase, ParseErrorInfo
from manim_rag.config import (
    DockerConfig,
    LLMConfig,
    LLMProvider,
    PipelineConfig,
    RetrievalConfig,
)
from manim_rag.execution import DockerSandboxExecutor, FailureClassifier
from manim_rag.generation import ManimGenerationPipeline, extract_python_code, extract_scene_name
from manim_rag.models import (
    AttemptRecord,
    ChunkType,
    CodeChunk,
    ExecutionResult,
    FailureCategory,
    FailureDiagnostic,
    PipelineResult,
    RetrievedChunk,
)
from manim_rag.orchestrator import ManimRAGOrchestrator
from manim_rag.retrieval import (
    BM25Index,
    ChromaVectorStore,
    DeterministicHashEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
)

__all__ = [
    "AttemptRecord",
    "BM25Index",
    "ChromaVectorStore",
    "ChunkType",
    "ChunkingResult",
    "CodeChunk",
    "DeterministicHashEmbedder",
    "DockerConfig",
    "DockerSandboxExecutor",
    "ExecutionResult",
    "FailureCategory",
    "FailureClassifier",
    "FailureDiagnostic",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LLMConfig",
    "LLMProvider",
    "ManimASTChunker",
    "ManimGenerationPipeline",
    "ManimKnowledgeBase",
    "ManimRAGOrchestrator",
    "ParseErrorInfo",
    "PipelineConfig",
    "PipelineResult",
    "RetrievalConfig",
    "RetrievedChunk",
    "extract_python_code",
    "extract_scene_name",
]
