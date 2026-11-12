"""LangChain generation pipeline for Llama 3 70B."""

from manim_rag.generation.pipeline import (
    ManimGenerationPipeline,
    build_llama3_chat_model,
    extract_python_code,
    extract_scene_name,
)
from manim_rag.generation.prompts import (
    INITIAL_GENERATION_PROMPT,
    REGENERATION_PROMPT,
    format_context,
    guidance_for_diagnostic,
)

__all__ = [
    "INITIAL_GENERATION_PROMPT",
    "ManimGenerationPipeline",
    "REGENERATION_PROMPT",
    "build_llama3_chat_model",
    "extract_python_code",
    "extract_scene_name",
    "format_context",
    "guidance_for_diagnostic",
]
