"""Tests for DockerSandboxExecutor command hardening, volume mounts, and timeouts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from manim_rag.config import DockerConfig
from manim_rag.execution import DockerSandboxExecutor


def test_build_command_security_flags_and_mounts(tmp_path: Path) -> None:
    executor = DockerSandboxExecutor(
        DockerConfig(
            image="manim-rag-sandbox:test",
            memory_limit="1g",
            cpus="1.5",
            pids_limit=64,
            quality="m",
        )
    )
    cmd = executor.build_command(
        host_workdir=tmp_path,
        scene_name="MyCircleScene",
        container_name="manim-rag-abc123",
    )

    assert cmd[0:3] == ["docker", "run", "--rm"]
    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges:true" in cmd
    assert "--pids-limit=64" in cmd
    assert "--memory=1g" in cmd
    assert "--cpus=1.5" in cmd
    assert f"{tmp_path.resolve()}:/workspace:rw" in cmd
    assert cmd[-6:] == [
        "manim",
        "-qm",
        "--media_dir",
        "/workspace/media",
        "/workspace/scene.py",
        "MyCircleScene",
    ]


def test_validate_scene_name_rejects_injection() -> None:
    with pytest.raises(ValueError):
        DockerSandboxExecutor.validate_scene_name("Scene; rm -rf /")
    with pytest.raises(ValueError):
        DockerSandboxExecutor.validate_scene_name("123BadName")


def test_execute_success_finds_rendered_mp4(tmp_path: Path) -> None:
    recorded_cmds: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append(cmd)
        video_file = tmp_path / "media" / "videos" / "scene" / "480p15" / "TestScene.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        video_file.write_bytes(b"fake-mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="Rendered TestScene", stderr="")

    executor = DockerSandboxExecutor(subprocess_runner=fake_runner)
    res = executor.execute(
        "from manim import *\nclass TestScene(Scene):\n    pass\n",
        scene_name="TestScene",
        workdir=tmp_path,
    )

    assert res.success is True
    assert res.exit_code == 0
    assert res.output_video_path is not None
    assert res.output_video_path.endswith("TestScene.mp4")
    assert (tmp_path / "scene.py").exists()
    assert len(recorded_cmds) == 1


def test_execute_timeout_triggers_container_cleanup(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def timeout_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[1] == "run":
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5, output="", stderr=b"")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    executor = DockerSandboxExecutor(
        DockerConfig(timeout_seconds=5),
        subprocess_runner=timeout_runner,
    )
    res = executor.execute(
        "from manim import *\nclass SlowScene(Scene):\n    pass\n",
        scene_name="SlowScene",
        workdir=tmp_path,
    )

    assert res.success is False
    assert res.timed_out is True
    assert res.exit_code == 124
    assert len(calls) == 2
    assert calls[1][:3] == ["docker", "rm", "-f"]


def test_execute_empty_code_fails_fast() -> None:
    executor = DockerSandboxExecutor()
    res = executor.execute("   \n  ", scene_name="EmptyScene")
    assert res.success is False
    assert "SyntaxError" in res.stderr
