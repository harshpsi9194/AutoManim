"""Sample ManimCE animation classes for AST indexing and RAG tests."""

from __future__ import annotations

from typing import Any


class Animation:
    """Base animation class."""

    def __init__(self, mobject: Any, run_time: float = 1.0, **kwargs: Any) -> None:
        self.mobject = mobject
        self.run_time = run_time


class Create(Animation):
    """Incrementally show a VMobject on screen (replaces deprecated ShowCreation)."""

    def __init__(
        self,
        mobject: Any,
        lag_ratio: float = 1.0,
        introducer: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(mobject, **kwargs)
        self.lag_ratio = lag_ratio


class FadeIn(Animation):
    """Fade in a mobject."""

    def __init__(
        self,
        *mobjects: Any,
        shift: Any | None = None,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(mobjects, **kwargs)


class Transform(Animation):
    """Morph one mobject into a target mobject."""

    def __init__(
        self,
        mobject: Any,
        target_mobject: Any | None = None,
        path_arc: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(mobject, **kwargs)
        self.target_mobject = target_mobject
