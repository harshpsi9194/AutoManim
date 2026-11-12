"""Tests for ManimCE traceback parsing and failure classification."""

from __future__ import annotations

from manim_rag.execution import FailureClassifier
from manim_rag.models import ExecutionResult, FailureCategory


def test_classify_api_errors() -> None:
    clf = FailureClassifier()

    attr_tb = """Traceback (most recent call last):
  File "/workspace/scene.py", line 6, in construct
    self.play(ShowCreation(Circle()))
AttributeError: module 'manim' has no attribute 'ShowCreation'
"""
    res = ExecutionResult(success=False, exit_code=1, stderr=attr_tb)
    diag = clf.classify(res, user_prompt="animate a circle")
    assert diag.category == FailureCategory.API_ERROR
    assert diag.error_type == "AttributeError"
    assert diag.line_number == 6
    assert "ShowCreation" in diag.extracted_symbols
    assert "ShowCreation" in diag.suggested_query

    name_tb = """Traceback (most recent call last):
  File "/workspace/scene.py", line 4, in construct
    txt = TextMobject("Hi")
NameError: name 'TextMobject' is not defined
"""
    diag_name = clf.classify(ExecutionResult(success=False, exit_code=1, stderr=name_tb))
    assert diag_name.category == FailureCategory.API_ERROR
    assert diag_name.error_type == "NameError"
    assert "TextMobject" in diag_name.extracted_symbols

    imp_tb = """Traceback (most recent call last):
  File "/workspace/scene.py", line 1, in <module>
    from manim import ShowCreation
ImportError: cannot import name 'ShowCreation' from 'manim'
"""
    diag_imp = clf.classify(ExecutionResult(success=False, exit_code=1, stderr=imp_tb))
    assert diag_imp.category == FailureCategory.API_ERROR
    assert diag_imp.error_type == "ImportError"
    assert "ShowCreation" in diag_imp.extracted_symbols


def test_classify_argument_errors_and_chained_tracebacks() -> None:
    clf = FailureClassifier()

    chained_tb = """Traceback (most recent call last):
  File "/workspace/scene.py", line 8, in construct
    c = Circle(radius_x=2.0)
  File "/usr/local/lib/python3.11/site-packages/manim/mobject/geometry/arc.py", line 310, in __init__
    super().__init__(**kwargs)
TypeError: Circle.__init__() got an unexpected keyword argument 'radius_x'
"""
    diag = clf.classify(ExecutionResult(success=False, exit_code=1, stderr=chained_tb))
    assert diag.category == FailureCategory.ARGUMENT_ERROR
    assert diag.error_type == "TypeError"
    # Should locate user's /workspace/scene.py line rather than manim internal site-packages line
    assert diag.line_number == 8
    assert "Circle" in diag.extracted_symbols
    assert "radius_x" in diag.extracted_symbols

    val_tb = """Traceback (most recent call last):
  File "/workspace/scene.py", line 10, in construct
    axes = Axes(x_range=(1,))
ValueError: x_range must have 2 or 3 elements
"""
    diag_val = clf.classify(ExecutionResult(success=False, exit_code=1, stderr=val_tb))
    assert diag_val.category == FailureCategory.ARGUMENT_ERROR
    assert diag_val.error_type == "ValueError"
    assert diag_val.line_number == 10


def test_classify_syntax_and_indentation_errors() -> None:
    clf = FailureClassifier()

    syntax_tb = """  File "/workspace/scene.py", line 12
    self.play(Create(circle)
              ^
SyntaxError: '(' was never closed
"""
    diag = clf.classify(ExecutionResult(success=False, exit_code=1, stderr=syntax_tb))
    assert diag.category == FailureCategory.SYNTAX_ERROR
    assert diag.error_type == "SyntaxError"
    assert diag.line_number == 12
    assert diag.offending_line is not None
    assert "self.play" in diag.offending_line
    assert diag.column_offset == 15

    indent_tb = """  File "/workspace/scene.py", line 5
    self.play(Create(c))
    ^
IndentationError: expected an indented block after function definition on line 4
"""
    diag_ind = clf.classify(ExecutionResult(success=False, exit_code=1, stderr=indent_tb))
    assert diag_ind.category == FailureCategory.SYNTAX_ERROR
    assert diag_ind.error_type == "IndentationError"
    assert diag_ind.line_number == 5


def test_classify_timeout_and_oom() -> None:
    clf = FailureClassifier()

    timeout_res = ExecutionResult(
        success=False,
        exit_code=124,
        timed_out=True,
        stderr="TimeoutError: Container timed out after 90s",
    )
    diag_t = clf.classify(timeout_res, user_prompt="render 3d fractal")
    assert diag_t.category == FailureCategory.TIMEOUT_ERROR
    assert diag_t.suggested_query == "render 3d fractal"

    oom_res = ExecutionResult(success=False, exit_code=137, stderr="")
    diag_oom = clf.classify(oom_res)
    assert diag_oom.category == FailureCategory.RUNTIME_ERROR
    assert diag_oom.error_type == "MemoryError"
