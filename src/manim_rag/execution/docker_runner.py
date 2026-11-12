"""Sandboxed Docker container executor for generated ManimCE scripts."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from manim_rag.config import DockerConfig
from manim_rag.models import ExecutionResult

_VALID_SCENE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


class DockerSandboxExecutor:
    """Runs generated ManimCE code inside an isolated Docker container."""

    def __init__(
        self,
        config: DockerConfig | None = None,
        *,
        subprocess_runner: SubprocessRunner | None = None,
    ) -> None:
        self.config = config or DockerConfig()
        self._runner: SubprocessRunner = subprocess_runner or subprocess.run

    @staticmethod
    def validate_scene_name(scene_name: str) -> str:
        """Ensure scene_name is a valid Python identifier before passing to Docker CLI."""
        if not _VALID_SCENE_RE.match(scene_name):
            raise ValueError(f"Invalid Manim scene name: {scene_name!r}")
        return scene_name

    def build_command(
        self,
        *,
        host_workdir: Path,
        scene_name: str,
        container_name: str,
        script_filename: str = "scene.py",
    ) -> list[str]:
        """Construct an argument-safe `docker run` command with strict sandbox flags."""
        safe_scene = self.validate_scene_name(scene_name)
        if "/" in script_filename or "\\" in script_filename or script_filename.startswith("."):
            raise ValueError(f"Invalid script filename: {script_filename!r}")

        resolved_workdir = str(host_workdir.resolve())
        c_dir = self.config.container_workdir.rstrip("/") or "/workspace"

        cmd: list[str] = [
            self.config.docker_binary,
            "run",
            "--rm",
            "--name",
            container_name,
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            f"--pids-limit={self.config.pids_limit}",
            f"--memory={self.config.memory_limit}",
            f"--cpus={self.config.cpus}",
            f"--user={self.config.user}",
            "-v",
            f"{resolved_workdir}:{c_dir}:rw",
            "-w",
            c_dir,
        ]

        if self.config.network_disabled:
            cmd.append("--network=none")

        if self.config.read_only_root:
            cmd.extend(
                [
                    "--read-only",
                    "--tmpfs",
                    f"/tmp:rw,noexec,nosuid,size={self.config.tmpfs_size}",
                ]
            )

        cmd.extend(
            [
                self.config.image,
                "manim",
                f"-q{self.config.quality}",
                "--media_dir",
                f"{c_dir}/media",
                f"{c_dir}/{script_filename}",
                safe_scene,
            ]
        )
        return cmd

    def execute(
        self,
        code: str,
        *,
        scene_name: str = "GeneratedScene",
        workdir: str | Path | None = None,
    ) -> ExecutionResult:
        """Write code to a workspace, run it inside Docker, and collect results."""
        safe_scene = self.validate_scene_name(scene_name)
        if not code.strip():
            return ExecutionResult(
                success=False,
                exit_code=1,
                stderr="SyntaxError: generated Manim script is empty",
            )

        if workdir is not None:
            run_dir = Path(workdir)
            run_dir.mkdir(parents=True, exist_ok=True)
            return self._execute_in_dir(code, safe_scene, run_dir)

        with tempfile.TemporaryDirectory(prefix="manim_rag_") as tmp_dir:
            return self._execute_in_dir(code, safe_scene, Path(tmp_dir))

    def _execute_in_dir(self, code: str, scene_name: str, run_dir: Path) -> ExecutionResult:
        script_path = run_dir / "scene.py"
        script_path.write_text(code, encoding="utf-8")
        (run_dir / "media").mkdir(exist_ok=True)

        container_name = f"manim-rag-{uuid.uuid4().hex[:12]}"
        cmd = self.build_command(
            host_workdir=run_dir,
            scene_name=scene_name,
            container_name=container_name,
        )

        start = time.monotonic()
        try:
            proc = self._runner(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
            duration = time.monotonic() - start
            output_video = self._find_output_video(run_dir / "media")
            return ExecutionResult(
                success=(proc.returncode == 0),
                exit_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=False,
                duration_seconds=duration,
                output_video_path=output_video,
                command=cmd,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            self._force_remove_container(container_name)
            stdout = _decode_maybe_bytes(exc.stdout)
            stderr = _decode_maybe_bytes(exc.stderr)
            if not stderr:
                stderr = f"TimeoutError: Container timed out after {self.config.timeout_seconds}s"
            return ExecutionResult(
                success=False,
                exit_code=124,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                duration_seconds=duration,
                command=cmd,
            )

    def _force_remove_container(self, container_name: str) -> None:
        try:
            self._runner(
                [self.config.docker_binary, "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            pass

    @staticmethod
    def _find_output_video(media_dir: Path) -> str | None:
        if not media_dir.exists():
            return None
        mp4s = sorted(media_dir.rglob("*.mp4"))
        if mp4s:
            return str(mp4s[-1])
        pngs = sorted(media_dir.rglob("*.png"))
        return str(pngs[-1]) if pngs else None


def _decode_maybe_bytes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
