"""LangChain prompt templates and ManimCE context formatting utilities."""

from __future__ import annotations

from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate

from manim_rag.models import CodeChunk, FailureCategory, FailureDiagnostic, RetrievedChunk

SYSTEM_PROMPT = """You are an expert Manim Community Edition (ManimCE) Python developer.
Write clean, self-contained, executable ManimCE scripts that fulfill the user's animation request.

Strict rules:
1. Always import from `manim` (`from manim import *`).
2. Define a single primary `Scene` subclass (e.g., `class GeneratedScene(Scene):` or `ThreeDScene` / `MovingCameraScene` when needed) with a `construct(self)` method.
3. Rely strictly on the retrieved ManimCE API signatures and conventions provided in the context. Do NOT invent deprecated ManimGL APIs such as `ShowCreation` (use `Create`), `TextMobject`/`TexMobject` (use `Text`/`MathTex`/`Tex`).
4. Output ONLY valid Python code inside a single ```python code block (or raw Python), with no trailing prose."""

INITIAL_USER_TEMPLATE = """### Retrieved ManimCE API Reference
{context}

### Animation Request
{user_prompt}

Generate the complete ManimCE Python script."""

REGENERATION_USER_TEMPLATE = """### Retrieved ManimCE API Reference (Updated for Error Diagnostics)
{context}

### Original Animation Request
{user_prompt}

### Previous Failing Code
```python
{previous_code}
```

### Execution Failure Diagnostics
- Category: {error_category}
- Error Type: {error_type}
- Line Number: {line_number}
- Message: {error_message}
- Analysis: {diagnostic_summary}
- Fix Guidance: {retry_guidance}

Fix the error and return the complete corrected ManimCE Python script."""

INITIAL_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", INITIAL_USER_TEMPLATE),
    ]
)

REGENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", REGENERATION_USER_TEMPLATE),
    ]
)

_CATEGORY_GUIDANCE: dict[FailureCategory, str] = {
    FailureCategory.API_ERROR: (
        "Replace missing/deprecated Manim symbols or attributes with valid ManimCE classes, "
        "methods, or imports from the retrieved reference above."
    ),
    FailureCategory.ARGUMENT_ERROR: (
        "Check exact parameter names, positional argument counts, and allowed values in the "
        "retrieved method/constructor signatures above."
    ),
    FailureCategory.SYNTAX_ERROR: (
        "Fix the Python syntax, indentation, or unclosed brackets/quotes around the reported line."
    ),
    FailureCategory.RUNTIME_ERROR: (
        "Resolve the runtime issue while keeping the scene mathematically and geometrically valid."
    ),
    FailureCategory.TIMEOUT_ERROR: (
        "Simplify expensive loops, reduce object counts or resolution-heavy operations so rendering "
        "completes quickly within the sandbox timeout."
    ),
    FailureCategory.UNKNOWN_ERROR: (
        "Review the traceback carefully and produce a simpler, verified ManimCE scene."
    ),
}


def guidance_for_diagnostic(diagnostic: FailureDiagnostic) -> str:
    """Return category-specific repair guidance for the regeneration prompt."""
    return _CATEGORY_GUIDANCE.get(
        diagnostic.category, _CATEGORY_GUIDANCE[FailureCategory.UNKNOWN_ERROR]
    )


def format_context(chunks: Sequence[RetrievedChunk | CodeChunk], *, max_chars: int = 12000) -> str:
    """Format retrieved ManimCE chunks into structured API reference blocks."""
    if not chunks:
        return "(No matching ManimCE API chunks retrieved.)"

    blocks: list[str] = []
    total_len = 0
    for idx, item in enumerate(chunks, start=1):
        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        header = f"[{idx}] {chunk.qualified_name} ({chunk.chunk_type.value}) — {chunk.module_path}"
        lines = [header]
        if chunk.signature:
            lines.append(f"Signature: {chunk.signature}")
        if chunk.docstring:
            doc_preview = chunk.docstring.strip()
            if len(doc_preview) > 400:
                doc_preview = doc_preview[:397] + "..."
            lines.append(f"Docstring: {doc_preview}")
        if chunk.source:
            src_preview = chunk.source.strip()
            if len(src_preview) > 1200:
                src_preview = src_preview[:1197] + "\n# ... (truncated)"
            lines.append(f"```python\n{src_preview}\n```")

        block = "\n".join(lines)
        if total_len + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total_len += len(block)

    return "\n\n".join(blocks)
