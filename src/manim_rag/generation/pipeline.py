"""LangChain LCEL generation pipeline targeting Llama 3 70B."""

from __future__ import annotations

import ast
import os
import re
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from manim_rag.config import LLMConfig, LLMProvider
from manim_rag.generation.prompts import (
    INITIAL_GENERATION_PROMPT,
    REGENERATION_PROMPT,
    format_context,
    guidance_for_diagnostic,
)
from manim_rag.models import CodeChunk, FailureDiagnostic, RetrievedChunk

_FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_SCENE_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*:", re.MULTILINE)


def extract_python_code(raw_output: str) -> str:
    """Extract clean Python source code from an LLM response."""
    text = raw_output.strip()
    matches = _FENCED_CODE_RE.findall(text)
    if matches:
        # Prefer the longest python fenced block if the model included multiple blocks.
        best = max(matches, key=len).strip()
        return best
    if text.startswith("```") and text.endswith("```"):
        inner = text.splitlines()[1:-1]
        return "\n".join(inner).strip()
    return text


def extract_scene_name(code: str) -> str | None:
    """Detect the Manim Scene subclass name from generated Python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        for match in _SCENE_CLASS_RE.finditer(code):
            class_name, bases = match.group(1), match.group(2)
            if "Scene" in bases:
                return class_name
        first = _SCENE_CLASS_RE.search(code)
        return first.group(1) if first else None

    fallback_class: str | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if fallback_class is None:
                fallback_class = node.name
            base_names = [ast.unparse(b) for b in node.bases]
            if any("Scene" in base for base in base_names):
                return node.name
    return fallback_class


def build_llama3_chat_model(config: LLMConfig) -> BaseChatModel:
    """Instantiate a LangChain ChatOpenAI client configured for Llama 3 70B."""
    from langchain_openai import ChatOpenAI

    env_var = config.resolved_api_key_env()
    api_key = config.api_key or os.environ.get(env_var)
    if not api_key:
        if config.provider in (LLMProvider.OLLAMA, LLMProvider.OPENAI_COMPATIBLE):
            api_key = "local-no-key-required"
        else:
            raise ValueError(
                f"API key required for provider '{config.provider.value}'. "
                f"Set LLMConfig.api_key or environment variable {env_var}."
            )

    return ChatOpenAI(
        model=config.resolved_model_name(),
        base_url=config.resolved_base_url(),
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout_seconds,
        max_retries=config.max_client_retries,
    )


class ManimGenerationPipeline:
    """LCEL pipeline for initial ManimCE generation and error-driven regeneration."""

    def __init__(
        self,
        *,
        config: LLMConfig | None = None,
        llm: Runnable[Any, Any] | BaseChatModel | None = None,
    ) -> None:
        self.config = config or LLMConfig()
        self._llm: Runnable[Any, Any] | BaseChatModel | None = llm
        self._parser = StrOutputParser()
        self._extractor = RunnableLambda(extract_python_code)

    @property
    def llm(self) -> Runnable[Any, Any] | BaseChatModel:
        """Lazily construct the Llama 3 70B LangChain chat model when needed."""
        if self._llm is None:
            self._llm = build_llama3_chat_model(self.config)
        return self._llm

    def build_initial_chain(self) -> Runnable[dict[str, Any], str]:
        """Build the LCEL chain for initial ManimCE code generation."""
        return INITIAL_GENERATION_PROMPT | self.llm | self._parser | self._extractor

    def build_regeneration_chain(self) -> Runnable[dict[str, Any], str]:
        """Build the LCEL chain for failure-aware code regeneration."""
        return REGENERATION_PROMPT | self.llm | self._parser | self._extractor

    def generate(
        self,
        user_prompt: str,
        retrieved_chunks: Sequence[RetrievedChunk | CodeChunk],
    ) -> str:
        """Generate an initial ManimCE script given a user prompt and retrieved API context."""
        chain = self.build_initial_chain()
        return chain.invoke(
            {
                "user_prompt": user_prompt,
                "context": format_context(retrieved_chunks),
            }
        )

    def regenerate(
        self,
        *,
        user_prompt: str,
        previous_code: str,
        diagnostic: FailureDiagnostic,
        retrieved_chunks: Sequence[RetrievedChunk | CodeChunk],
    ) -> str:
        """Regenerate ManimCE code using classified failure diagnostics and targeted context."""
        chain = self.build_regeneration_chain()
        return chain.invoke(
            {
                "user_prompt": user_prompt,
                "previous_code": previous_code,
                "context": format_context(retrieved_chunks),
                "error_category": diagnostic.category.value,
                "error_type": diagnostic.error_type,
                "error_message": diagnostic.error_message,
                "line_number": (
                    str(diagnostic.line_number) if diagnostic.line_number is not None else "unknown"
                ),
                "diagnostic_summary": diagnostic.diagnostic_summary,
                "retry_guidance": guidance_for_diagnostic(diagnostic),
            }
        )
