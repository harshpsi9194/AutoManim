"""Traceback parser and failure classifier for ManimCE execution diagnostics."""

from __future__ import annotations

import re

from manim_rag.models import ExecutionResult, FailureCategory, FailureDiagnostic

_FILE_LINE_RE = re.compile(r'^\s*File\s+"([^"]+)",\s+line\s+(\d+)(?:,\s+in\s+(\S+))?')
_ERROR_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*:\s*(.*)$")
_BARE_ERROR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*$")

_API_ERROR_TYPES = {
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
    "UnboundLocalError",
}
_ARGUMENT_ERROR_TYPES = {
    "TypeError",
    "ValueError",
}
_SYNTAX_ERROR_TYPES = {
    "SyntaxError",
    "IndentationError",
    "TabError",
}

_SYMBOL_PATTERNS = [
    re.compile(r"module\s+'([^']+)'\s+has no attribute\s+'([^']+)'"),
    re.compile(r"'([^']+)'\s+object has no attribute\s+'([^']+)'"),
    re.compile(r"type object\s+'([^']+)'\s+has no attribute\s+'([^']+)'"),
    re.compile(r"cannot import name\s+'([^']+)'\s+from\s+'([^']+)'"),
    re.compile(r"No module named\s+'([^']+)'"),
    re.compile(r"name\s+'([^']+)'\s+is not defined"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\(\)\s+got an unexpected keyword argument\s+'([^']+)'"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\(\)\s+missing \d+ required (?:positional|keyword-only) argument[s]?:\s*(.+)"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\(\)\s+takes\s+.*argument"),
]


def _extract_symbols(error_message: str) -> list[str]:
    symbols: list[str] = []
    for pattern in _SYMBOL_PATTERNS:
        match = pattern.search(error_message)
        if not match:
            continue
        for grp in match.groups():
            for piece in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", grp):
                cleaned = piece.removesuffix(".__init__")
                if cleaned and cleaned not in symbols:
                    symbols.append(cleaned)
                if "." in cleaned:
                    tail = cleaned.split(".")[-1]
                    if tail and tail not in symbols:
                        symbols.append(tail)
    return symbols


def _find_error_line(lines: list[str]) -> tuple[str, str, int]:
    """Scan from bottom up to find the final Python exception type, message, and line index."""
    for idx in range(len(lines) - 1, -1, -1):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        match = _ERROR_LINE_RE.match(stripped)
        if match:
            return match.group(1), match.group(2).strip(), idx
        bare = _BARE_ERROR_RE.match(stripped)
        if bare:
            return bare.group(1), "", idx
    return "UnknownError", "", len(lines) - 1


def _find_user_frame(
    lines: list[str],
    limit_idx: int,
    script_names: tuple[str, ...] = ("scene.py", "/workspace/scene.py"),
) -> tuple[int | None, str | None, int | None]:
    """Extract (line_number, offending_line, column_offset) preferring the generated script."""
    frames: list[tuple[str, int, int]] = []
    for idx in range(min(limit_idx + 1, len(lines))):
        match = _FILE_LINE_RE.match(lines[idx])
        if match:
            frames.append((match.group(1), int(match.group(2)), idx))

    if not frames:
        return None, None, None

    chosen = frames[-1]
    for frame in reversed(frames):
        if any(frame[0].endswith(name) or frame[0] == name for name in script_names):
            chosen = frame
            break

    _, line_num, frame_idx = chosen
    offending_line: str | None = None
    col_offset: int | None = None

    if frame_idx + 1 < limit_idx:
        candidate = lines[frame_idx + 1]
        if not _FILE_LINE_RE.match(candidate) and not _ERROR_LINE_RE.match(candidate.strip()):
            offending_line = candidate.rstrip("\n")
    if frame_idx + 2 < limit_idx:
        caret_line = lines[frame_idx + 2]
        if "^" in caret_line or "~" in caret_line:
            pos = caret_line.find("^")
            if pos >= 0:
                col_offset = pos + 1

    return line_num, offending_line, col_offset


def _classify_error_type(error_type: str) -> FailureCategory:
    short_type = error_type.split(".")[-1]
    if short_type in _SYNTAX_ERROR_TYPES:
        return FailureCategory.SYNTAX_ERROR
    if short_type in _API_ERROR_TYPES:
        return FailureCategory.API_ERROR
    if short_type in _ARGUMENT_ERROR_TYPES:
        return FailureCategory.ARGUMENT_ERROR
    if short_type in {"TimeoutError", "TimeoutExpired"}:
        return FailureCategory.TIMEOUT_ERROR
    if short_type != "UnknownError":
        return FailureCategory.RUNTIME_ERROR
    return FailureCategory.UNKNOWN_ERROR


class FailureClassifier:
    """Classifies ManimCE execution errors into API, Argument, Syntax, or Runtime failures."""

    def classify(
        self,
        execution: ExecutionResult,
        *,
        user_prompt: str = "",
    ) -> FailureDiagnostic:
        """Analyze stderr/stdout/exit_code and return structured FailureDiagnostic."""
        if execution.timed_out or execution.exit_code == 124:
            return FailureDiagnostic(
                category=FailureCategory.TIMEOUT_ERROR,
                error_type="TimeoutError",
                error_message="Docker sandbox execution exceeded timeout limit.",
                suggested_query=user_prompt,
                diagnostic_summary=(
                    "Execution timed out inside the sandboxed container. Simplify scene geometry "
                    "or animation complexity."
                ),
                raw_traceback=(execution.stderr or execution.stdout).strip(),
            )

        combined = "\n".join(part for part in (execution.stderr, execution.stdout) if part).strip()
        if not combined:
            if execution.exit_code == 137:
                return FailureDiagnostic(
                    category=FailureCategory.RUNTIME_ERROR,
                    error_type="MemoryError",
                    error_message="Container killed (exit code 137 / OOM).",
                    suggested_query=user_prompt,
                    diagnostic_summary="Container exceeded memory limit and was killed (SIGKILL).",
                )
            return FailureDiagnostic(
                category=FailureCategory.UNKNOWN_ERROR,
                error_type="UnknownError",
                error_message=f"Execution failed with exit code {execution.exit_code} and no output.",
                suggested_query=user_prompt,
                diagnostic_summary=f"Process exited with code {execution.exit_code}.",
            )

        lines = combined.splitlines()
        error_type, error_message, err_idx = _find_error_line(lines)
        category = _classify_error_type(error_type)
        line_number, offending_line, col_offset = _find_user_frame(lines, err_idx)
        extracted_symbols = _extract_symbols(error_message)

        if category == FailureCategory.UNKNOWN_ERROR and execution.exit_code == 137:
            category = FailureCategory.RUNTIME_ERROR
            error_type = "MemoryError"
            error_message = error_message or "Container killed (exit code 137)."

        suggested_query = self._build_suggested_query(
            category=category,
            error_type=error_type,
            extracted_symbols=extracted_symbols,
            user_prompt=user_prompt,
        )
        summary = self._build_summary(
            category=category,
            error_type=error_type,
            error_message=error_message,
            line_number=line_number,
            extracted_symbols=extracted_symbols,
        )

        return FailureDiagnostic(
            category=category,
            error_type=error_type,
            error_message=error_message or combined.splitlines()[-1],
            line_number=line_number,
            column_offset=col_offset,
            offending_line=offending_line,
            extracted_symbols=extracted_symbols,
            suggested_query=suggested_query,
            diagnostic_summary=summary,
            raw_traceback=combined,
        )

    @staticmethod
    def _build_suggested_query(
        *,
        category: FailureCategory,
        error_type: str,
        extracted_symbols: list[str],
        user_prompt: str,
    ) -> str:
        parts: list[str] = []
        if extracted_symbols:
            parts.extend(extracted_symbols)
        if category == FailureCategory.API_ERROR:
            parts.append("manim class method api")
        elif category == FailureCategory.ARGUMENT_ERROR:
            parts.append("signature parameters kwargs")
        if user_prompt:
            parts.append(user_prompt)
        if not parts:
            return error_type
        return " ".join(parts)

    @staticmethod
    def _build_summary(
        *,
        category: FailureCategory,
        error_type: str,
        error_message: str,
        line_number: int | None,
        extracted_symbols: list[str],
    ) -> str:
        loc = f" at line {line_number}" if line_number is not None else ""
        syms = f" (symbols: {', '.join(extracted_symbols)})" if extracted_symbols else ""
        return f"[{category.value}] {error_type}{loc}: {error_message}{syms}".strip()
