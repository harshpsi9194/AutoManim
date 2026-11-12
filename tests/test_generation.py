"""Tests for the LangChain Llama 3 70B generation and prompt pipeline."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from manim_rag.config import LLMConfig, LLMProvider
from manim_rag.generation import (
    ManimGenerationPipeline,
    build_llama3_chat_model,
    extract_python_code,
    extract_scene_name,
    format_context,
)
from manim_rag.models import FailureCategory, FailureDiagnostic
from manim_rag.retrieval import HybridRetriever


def test_llm_config_provider_presets_and_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    groq = LLMConfig(provider=LLMProvider.GROQ)
    assert "70b" in groq.resolved_model_name().lower()
    assert "groq.com" in groq.resolved_base_url()

    together = LLMConfig(provider=LLMProvider.TOGETHER)
    assert "Llama-3-70b" in together.resolved_model_name()
    assert "together.xyz" in together.resolved_base_url()

    ollama = LLMConfig(provider=LLMProvider.OLLAMA)
    assert ollama.resolved_model_name() == "llama3:70b"
    assert "11434" in ollama.resolved_base_url()

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_llama3_chat_model(groq)

    # Ollama allows local usage without a real API key.
    ollama_model = build_llama3_chat_model(ollama)
    assert ollama_model is not None


def test_extract_python_code_and_scene_name() -> None:
    raw = """Here is your scene:
```python
from manim import *

class CircleToSquare(Scene):
    def construct(self):
        c = Circle()
        self.play(Create(c))
```
Hope this helps!"""
    code = extract_python_code(raw)
    assert code.startswith("from manim import *")
    assert "```" not in code
    assert extract_scene_name(code) == "CircleToSquare"

    # Syntax-invalid code falls back to regex Scene detection.
    broken = "from manim import *\nclass BrokenThreeD(ThreeDScene):\n    def construct(self:\n"
    assert extract_scene_name(broken) == "BrokenThreeD"


def test_lcel_generation_and_regeneration_with_fake_chat_model(
    sample_retriever: HybridRetriever,
) -> None:
    chunks = sample_retriever.retrieve("Create Circle", top_k=3)
    formatted = format_context(chunks)
    assert "Circle" in formatted

    captured_prompts: list[str] = []

    fake = FakeListChatModel(
        responses=[
            "```python\nfrom manim import *\nclass DrawCircle(Scene):\n    def construct(self):\n        self.play(Create(Circle()))\n```",
        ]
    )

    def _inspect_and_forward(prompt_value: Any) -> str:
        messages: list[BaseMessage] = prompt_value.to_messages()
        captured_prompts.append("\n".join(str(m.content) for m in messages))
        return str(fake.invoke(prompt_value).content)

    pipeline = ManimGenerationPipeline(llm=RunnableLambda(_inspect_and_forward))
    code = pipeline.generate("Draw a circle", chunks)
    assert "class DrawCircle(Scene):" in code
    assert "Retrieved ManimCE API Reference" in captured_prompts[0]
    assert "Draw a circle" in captured_prompts[0]

    # Test regeneration injects diagnostics and previous failing code.
    fake_regen = FakeListChatModel(
        responses=[
            "```python\nfrom manim import *\nclass DrawCircle(Scene):\n    def construct(self):\n        self.play(Create(Circle(radius=2)))\n```"
        ]
    )
    regen_prompts: list[str] = []

    def _inspect_regen(prompt_value: Any) -> str:
        messages = prompt_value.to_messages()
        regen_prompts.append("\n".join(str(m.content) for m in messages))
        return str(fake_regen.invoke(prompt_value).content)

    regen_pipeline = ManimGenerationPipeline(llm=RunnableLambda(_inspect_regen))
    diag = FailureDiagnostic(
        category=FailureCategory.API_ERROR,
        error_type="AttributeError",
        error_message="module 'manim' has no attribute 'ShowCreation'",
        line_number=5,
        extracted_symbols=["manim", "ShowCreation"],
        diagnostic_summary="[api_error] AttributeError at line 5",
    )
    fixed = regen_pipeline.regenerate(
        user_prompt="Draw a circle",
        previous_code="self.play(ShowCreation(Circle()))",
        diagnostic=diag,
        retrieved_chunks=chunks,
    )
    assert "Create(Circle(radius=2))" in fixed
    assert "ShowCreation" in regen_prompts[0]
    assert "api_error" in regen_prompts[0]
    assert "line_number" not in regen_prompts[0]  # template placeholder must be filled
    assert "Line Number: 5" in regen_prompts[0]
