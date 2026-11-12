"""End-to-end RAG-execute-classify-regenerate orchestrator for ManimCE."""

from __future__ import annotations

from pathlib import Path

from manim_rag.config import PipelineConfig
from manim_rag.execution.classifier import FailureClassifier
from manim_rag.execution.docker_runner import DockerSandboxExecutor
from manim_rag.generation.pipeline import ManimGenerationPipeline, extract_scene_name
from manim_rag.models import (
    AttemptRecord,
    ExecutionResult,
    FailureCategory,
    FailureDiagnostic,
    PipelineResult,
    RetrievedChunk,
)
from manim_rag.retrieval.hybrid import HybridRetriever


def _merge_chunks(
    primary: list[RetrievedChunk],
    targeted: list[RetrievedChunk],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    """Merge targeted retry chunks ahead of prior context while deduplicating by chunk_id."""
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for item in [*targeted, *primary]:
        if item.chunk.chunk_id in seen:
            continue
        seen.add(item.chunk.chunk_id)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


class ManimRAGOrchestrator:
    """Coordinates retrieval, Llama 3 70B generation, Docker execution, and retry repair."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        generator: ManimGenerationPipeline,
        executor: DockerSandboxExecutor,
        classifier: FailureClassifier | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.executor = executor
        self.classifier = classifier or FailureClassifier()
        self.config = config or PipelineConfig()

    def run(
        self,
        user_prompt: str,
        *,
        workdir: str | Path | None = None,
    ) -> PipelineResult:
        """Generate, execute, classify failures, and regenerate up to max_retries."""
        initial_query = user_prompt
        base_chunks = self.retriever.retrieve(initial_query, top_k=self.config.retrieval.top_k)
        active_chunks = list(base_chunks)
        current_query = initial_query

        attempts: list[AttemptRecord] = []
        last_code = ""
        last_scene: str | None = None
        last_diagnostic: FailureDiagnostic | None = None

        max_attempts = 1 + self.config.max_retries
        for attempt_idx in range(1, max_attempts + 1):
            if attempt_idx == 1:
                code = self.generator.generate(user_prompt, active_chunks)
            else:
                assert last_diagnostic is not None
                code = self.generator.regenerate(
                    user_prompt=user_prompt,
                    previous_code=last_code,
                    diagnostic=last_diagnostic,
                    retrieved_chunks=active_chunks,
                )

            scene_name = extract_scene_name(code) or "GeneratedScene"
            last_code = code
            last_scene = scene_name

            if not code.strip():
                execution = ExecutionResult(
                    success=False,
                    exit_code=1,
                    stderr="SyntaxError: LLM returned empty code",
                )
            else:
                execution = self.executor.execute(code, scene_name=scene_name, workdir=workdir)

            if execution.success:
                attempts.append(
                    AttemptRecord(
                        attempt_number=attempt_idx,
                        prompt_used=user_prompt,
                        retrieved_chunk_ids=[c.chunk.chunk_id for c in active_chunks],
                        retrieval_query=current_query,
                        generated_code=code,
                        scene_name=scene_name,
                        execution=execution,
                        diagnostic=None,
                    )
                )
                return PipelineResult(
                    success=True,
                    user_prompt=user_prompt,
                    final_code=code,
                    scene_name=scene_name,
                    output_video_path=execution.output_video_path,
                    total_attempts=attempt_idx,
                    attempts=attempts,
                    last_diagnostic=None,
                )

            diagnostic = self.classifier.classify(execution, user_prompt=user_prompt)
            last_diagnostic = diagnostic
            attempts.append(
                AttemptRecord(
                    attempt_number=attempt_idx,
                    prompt_used=user_prompt,
                    retrieved_chunk_ids=[c.chunk.chunk_id for c in active_chunks],
                    retrieval_query=current_query,
                    generated_code=code,
                    scene_name=scene_name,
                    execution=execution,
                    diagnostic=diagnostic,
                )
            )

            if attempt_idx < max_attempts:
                current_query = diagnostic.suggested_query or user_prompt
                targeted = self.retriever.retrieve(
                    current_query,
                    top_k=self.config.retry_top_k,
                )
                # For API/Argument errors prioritize newly retrieved symbol docs; keep prior context too.
                if diagnostic.category in (
                    FailureCategory.API_ERROR,
                    FailureCategory.ARGUMENT_ERROR,
                ):
                    active_chunks = _merge_chunks(
                        base_chunks,
                        targeted,
                        limit=self.config.retrieval.top_k + self.config.retry_top_k,
                    )
                else:
                    active_chunks = _merge_chunks(
                        targeted,
                        base_chunks,
                        limit=self.config.retrieval.top_k,
                    )

        return PipelineResult(
            success=False,
            user_prompt=user_prompt,
            final_code=last_code,
            scene_name=last_scene,
            output_video_path=None,
            total_attempts=len(attempts),
            attempts=attempts,
            last_diagnostic=last_diagnostic,
        )
