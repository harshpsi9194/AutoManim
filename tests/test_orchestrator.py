"""End-to-end integration tests for ManimRAGOrchestrator and retry loop."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from manim_rag.config import PipelineConfig
from manim_rag.execution import DockerSandboxExecutor, FailureClassifier
from manim_rag.generation import ManimGenerationPipeline
from manim_rag.models import FailureCategory
from manim_rag.orchestrator import ManimRAGOrchestrator
from manim_rag.retrieval import HybridRetriever


def test_orchestrator_recovers_from_syntax_and_api_errors(
    sample_retriever: HybridRetriever, tmp_path: Path
) -> None:
    # Attempt 1: SyntaxError
    bad_syntax = "```python\nfrom manim import *\nclass MyScene(Scene):\n    def construct(self):\n        self.play(Create(Circle()\n```"
    # Attempt 2: Deprecated ManimGL API ShowCreation -> AttributeError
    bad_api = "```python\nfrom manim import *\nclass MyScene(Scene):\n    def construct(self):\n        self.play(ShowCreation(Circle()))\n```"
    # Attempt 3: Valid ManimCE code
    valid_code = "```python\nfrom manim import *\nclass MyScene(Scene):\n    def construct(self):\n        self.play(Create(Circle()))\n```"

    fake_llm = FakeListChatModel(responses=[bad_syntax, bad_api, valid_code])

    def fake_docker(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        script = (tmp_path / "scene.py").read_text(encoding="utf-8")
        if "Create(Circle()\n" in script or script.count("(") != script.count(")"):
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=(
                    '  File "/workspace/scene.py", line 4\n'
                    "        self.play(Create(Circle()\n"
                    "                  ^\n"
                    "SyntaxError: '(' was never closed\n"
                ),
            )
        if "ShowCreation" in script:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=(
                    "Traceback (most recent call last):\n"
                    '  File "/workspace/scene.py", line 4, in construct\n'
                    "    self.play(ShowCreation(Circle()))\n"
                    "AttributeError: module 'manim' has no attribute 'ShowCreation'\n"
                ),
            )
        video = tmp_path / "media" / "videos" / "scene" / "480p15" / "MyScene.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="Success", stderr="")

    orchestrator = ManimRAGOrchestrator(
        retriever=sample_retriever,
        generator=ManimGenerationPipeline(llm=fake_llm),
        executor=DockerSandboxExecutor(subprocess_runner=fake_docker),
        classifier=FailureClassifier(),
        config=PipelineConfig(max_retries=3),
    )

    result = orchestrator.run("Draw a circle on screen", workdir=tmp_path)

    assert result.success is True
    assert result.total_attempts == 3
    assert result.scene_name == "MyScene"
    assert result.output_video_path is not None
    assert result.attempts[0].diagnostic is not None
    assert result.attempts[0].diagnostic.category == FailureCategory.SYNTAX_ERROR
    assert result.attempts[1].diagnostic is not None
    assert result.attempts[1].diagnostic.category == FailureCategory.API_ERROR
    assert "ShowCreation" in result.attempts[2].retrieval_query
    assert result.attempts[2].diagnostic is None


def test_orchestrator_stops_at_max_retries_and_handles_zero_retries(
    sample_retriever: HybridRetriever, tmp_path: Path
) -> None:
    bad_arg = "```python\nfrom manim import *\nclass ArgScene(Scene):\n    def construct(self):\n        c = Circle(invalid_kw=1)\n```"
    fake_llm = FakeListChatModel(responses=[bad_arg, bad_arg, bad_arg])

    def always_arg_error(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                '  File "/workspace/scene.py", line 4, in construct\n'
                "    c = Circle(invalid_kw=1)\n"
                "TypeError: Circle.__init__() got an unexpected keyword argument 'invalid_kw'\n"
            ),
        )

    orchestrator = ManimRAGOrchestrator(
        retriever=sample_retriever,
        generator=ManimGenerationPipeline(llm=fake_llm),
        executor=DockerSandboxExecutor(subprocess_runner=always_arg_error),
        config=PipelineConfig(max_retries=2),
    )
    res = orchestrator.run("Create a circle", workdir=tmp_path)
    assert res.success is False
    assert res.total_attempts == 3  # 1 initial + 2 retries
    assert res.last_diagnostic is not None
    assert res.last_diagnostic.category == FailureCategory.ARGUMENT_ERROR

    # Zero retries should run exactly 1 attempt.
    zero_orch = ManimRAGOrchestrator(
        retriever=sample_retriever,
        generator=ManimGenerationPipeline(llm=FakeListChatModel(responses=[bad_arg])),
        executor=DockerSandboxExecutor(subprocess_runner=always_arg_error),
        config=PipelineConfig(max_retries=0),
    )
    zero_res = zero_orch.run("Create a circle", workdir=tmp_path)
    assert zero_res.success is False
    assert zero_res.total_attempts == 1
