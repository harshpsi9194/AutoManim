"""Code-aware BM25 keyword search over AST-parsed ManimCE chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence

from manim_rag.models import CodeChunk

try:
    from rank_bm25 import BM25Okapi as _RankBM25Okapi
except ImportError:  # pragma: no cover - fallback tested independently
    _RankBM25Okapi = None

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize_code(text: str) -> list[str]:
    """Tokenize Python code and docstrings preserving exact and split identifiers."""
    tokens: list[str] = []
    for raw in _IDENT_RE.findall(text):
        lower = raw.lower()
        tokens.append(lower)
        parts = [p for p in raw.split("_") if p]
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
        for part in parts:
            camel_parts = _CAMEL_SPLIT_RE.split(part)
            if len(camel_parts) > 1:
                tokens.extend(c.lower() for c in camel_parts if c)
    return tokens


class _PurePythonBM25:
    """Lightweight Okapi BM25 fallback when rank_bm25 is unavailable."""

    def __init__(self, corpus: Sequence[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size else 0.0

        df: Counter[str] = Counter()
        for counts in self.doc_freqs:
            for term in counts:
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query: Sequence[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0 or self.avgdl == 0.0:
            return scores
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, counts in enumerate(self.doc_freqs):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / self.avgdl))
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


def _matches_filter(chunk: CodeChunk, where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    meta = chunk.to_flat_metadata()
    for key, expected in where.items():
        if meta.get(key) != expected:
            return False
    return True


class BM25Index:
    """BM25 keyword index over CodeChunks."""

    def __init__(self, chunks: Sequence[CodeChunk] | None = None) -> None:
        self._chunks: list[CodeChunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: Any = None
        if chunks:
            self.index_chunks(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def index_chunks(self, chunks: Sequence[CodeChunk]) -> None:
        """Build or rebuild the BM25 index from a sequence of CodeChunks."""
        self._chunks = list(chunks)
        self._tokenized = [tokenize_code(chunk.to_document_text()) for chunk in self._chunks]
        if not self._chunks:
            self._bm25 = None
            return
        if _RankBM25Okapi is not None:
            self._bm25 = _RankBM25Okapi(self._tokenized)
        else:  # pragma: no cover
            self._bm25 = _PurePythonBM25(self._tokenized)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """Search indexed chunks using BM25 scores."""
        if not self._chunks or self._bm25 is None or top_k <= 0:
            return []
        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scored: list[tuple[CodeChunk, float]] = []
        for chunk, score in zip(self._chunks, raw_scores):
            f_score = float(score)
            if f_score <= 0.0:
                continue
            if not _matches_filter(chunk, where):
                continue
            scored.append((chunk, f_score))

        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return scored[:top_k]
