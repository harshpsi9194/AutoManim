"""Sample ManimCE geometry mobjects for AST indexing and RAG tests."""

from __future__ import annotations

from typing import Any


class Mobject:
    """Base mathematical object in ManimCE."""

    def set_color(self, color: str = "WHITE", family: bool = True) -> "Mobject":
        """Set the color of the mobject."""
        return self

    def shift(self, *vectors: Any) -> "Mobject":
        """Shift the mobject by one or more vectors."""
        return self

    def scale(self, scale_factor: float, **kwargs: Any) -> "Mobject":
        """Scale the size of the mobject."""
        return self


class Circle(Mobject):
    """A circular arc closed into a circle."""

    def __init__(
        self,
        radius: float = 1.0,
        color: str = "RED",
        fill_opacity: float = 0.0,
        stroke_width: float = 4.0,
        **kwargs: Any,
    ) -> None:
        self.radius = radius
        self.color = color
        self.fill_opacity = fill_opacity
        self.stroke_width = stroke_width

    def surround(self, mobject: Mobject, buffer_factor: float = 1.2) -> "Circle":
        """Resize the circle so it surrounds the given mobject."""
        return self


class Square(Mobject):
    """A square with equal side lengths."""

    def __init__(
        self,
        side_length: float = 2.0,
        color: str = "WHITE",
        fill_opacity: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.side_length = side_length
        self.color = color
        self.fill_opacity = fill_opacity


class Axes(Mobject):
    """Creates a set of coordinate axes."""

    def __init__(
        self,
        x_range: tuple[float, float, float] = (-5, 5, 1),
        y_range: tuple[float, float, float] = (-3, 3, 1),
        x_length: float | None = None,
        y_length: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range

    def plot(
        self,
        function: Any,
        x_range: tuple[float, float, float] | None = None,
        color: str = "BLUE",
        use_smoothing: bool = True,
    ) -> Mobject:
        """Plot a 2D curve y = function(x) on the axes."""
        return Mobject()
