"""Embedding providers for vector indexing and local deterministic testing."""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _subtokenize(text: str) -> list[str]:
    """Extract whole identifiers, snake/camel sub-tokens, and char 3-grams."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lower_raw = raw.lower()
        tokens.append(lower_raw)
        for part in raw.split("_"):
            if not part:
                continue
            for sub in _CAMEL_RE.split(part):
                sub_l = sub.lower()
                if len(sub_l) >= 2:
                    tokens.append(sub_l)
            if len(part) >= 4:
                p_low = part.lower()
                for idx in range(len(p_low) - 2):
                    tokens.append(f"#{p_low[idx : idx + 3]}")
    return tokens


class EmbeddingProvider(Protocol):
    """Protocol for text-to-vector embedding backends."""

    @property
    def dimension(self) -> int:
        """Dimensionality of output vectors."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        ...


class DeterministicHashEmbedder:
    """Zero-dependency local feature-hashing embedder for tests and offline runs."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dimension, dtype=np.float64)
        tokens = _subtokenize(text)
        if not tokens:
            return vec.tolist()

        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            idx = val % self._dimension
            sign = 1.0 if ((val >> 1) & 1) == 0 else -1.0
            weight = 1.5 if not tok.startswith("#") else 0.5
            vec[idx] += sign * weight

        norm = float(math.sqrt(float(np.dot(vec, vec))))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)


class OpenAICompatibleEmbedder:
    """Embedding provider using an OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        dimension: int = 1536,
    ) -> None:
        from langchain_openai import OpenAIEmbeddings

        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise ValueError(
                f"Missing API key for OpenAICompatibleEmbedder (checked {api_key_env})."
            )
        kwargs: dict[str, object] = {"model": model, "api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAIEmbeddings(**kwargs)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._client.embed_documents(list(texts))

    def embed_query(self, query: str) -> list[float]:
        return self._client.embed_query(query)
