"""Configuration models and presets for retrieval, Llama 3 70B LLM, and Docker sandbox."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported OpenAI-compatible and local providers for Llama 3 70B."""

    GROQ = "groq"
    TOGETHER = "together"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI_COMPATIBLE = "openai_compatible"


_PROVIDER_DEFAULTS: dict[LLMProvider, tuple[str, str, str]] = {
    LLMProvider.GROQ: (
        "llama-3.3-70b-versatile",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
    ),
    LLMProvider.TOGETHER: (
        "meta-llama/Llama-3-70b-chat-hf",
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
    ),
    LLMProvider.OLLAMA: (
        "llama3:70b",
        "http://localhost:11434/v1",
        "OLLAMA_API_KEY",
    ),
    LLMProvider.HUGGINGFACE: (
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "https://api-inference.huggingface.co/v1",
        "HF_TOKEN",
    ),
    LLMProvider.OPENAI_COMPATIBLE: (
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "http://localhost:8000/v1",
        "OPENAI_API_KEY",
    ),
}


class LLMConfig(BaseModel):
    """Configuration for the Llama 3 70B LangChain chat model."""

    provider: LLMProvider = LLMProvider.GROQ
    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_client_retries: int = Field(default=2, ge=0)

    def resolved_model_name(self) -> str:
        """Return explicit model name or provider's Llama 3 70B default."""
        return self.model_name or _PROVIDER_DEFAULTS[self.provider][0]

    def resolved_base_url(self) -> str:
        """Return explicit base URL or provider default."""
        return self.base_url or _PROVIDER_DEFAULTS[self.provider][1]

    def resolved_api_key_env(self) -> str:
        """Return explicit API key env var name or provider default."""
        return self.api_key_env or _PROVIDER_DEFAULTS[self.provider][2]


class RetrievalConfig(BaseModel):
    """Configuration for hybrid vector + BM25 retrieval."""

    top_k: int = Field(default=6, gt=0)
    vector_candidates: int = Field(default=15, gt=0)
    bm25_candidates: int = Field(default=15, gt=0)
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding_dim: int = Field(default=256, gt=0)
    collection_name: str = "manimce_knowledge_base"


class DockerConfig(BaseModel):
    """Security and resource configuration for sandboxed Docker execution."""

    image: str = "manim-rag-sandbox:latest"
    timeout_seconds: int = Field(default=90, gt=0)
    memory_limit: str = "2g"
    cpus: str = "2.0"
    pids_limit: int = Field(default=128, gt=0)
    network_disabled: bool = True
    read_only_root: bool = True
    tmpfs_size: str = "256m"
    quality: Literal["l", "m", "h", "p", "k"] = "l"
    user: str = "1000:1000"
    container_workdir: str = "/workspace"
    docker_binary: str = "docker"


class PipelineConfig(BaseModel):
    """Top-level configuration for the Manim RAG generation & repair loop."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    max_retries: int = Field(default=3, ge=0)
    retry_top_k: int = Field(default=4, gt=0)
