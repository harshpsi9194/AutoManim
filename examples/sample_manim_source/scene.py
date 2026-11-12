"""Sample ManimCE Scene API definitions."""

from __future__ import annotations

from typing import Any


class Scene:
    """The canvas of an animation in ManimCE."""

    def construct(self) -> None:
        """Override construct to define animations."""

    def play(self, *args: Any, run_time: float | None = None, **kwargs: Any) -> None:
        """Play one or more animations concurrently."""

    def wait(self, duration: float = 1.0) -> None:
        """Pause the scene for duration seconds."""

    def add(self, *mobjects: Any) -> "Scene":
        """Add mobjects immediately without animating."""
        return self
