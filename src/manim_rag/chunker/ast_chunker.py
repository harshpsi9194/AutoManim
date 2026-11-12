"""Python AST-based source chunker for ManimCE API extraction."""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from manim_rag.models import ChunkType, CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class ParseErrorInfo:
    """Details about a file that failed AST parsing."""

    file_path: str
    error_type: str
    message: str
    line_number: int | None = None


@dataclass
class ChunkingResult:
    """Aggregated output of parsing one or more Python files."""

    chunks: list[CodeChunk] = field(default_factory=list)
    errors: list[ParseErrorInfo] = field(default_factory=list)
    files_processed: int = 0


def _unparse_safe(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def _format_arguments(args: ast.arguments) -> str:
    """Reconstruct a clean parameter list string from an ast.arguments node."""
    parts: list[str] = []

    pos_args = list(args.posonlyargs) + list(args.args)
    num_pos = len(pos_args)
    num_defaults = len(args.defaults)
    default_offset = num_pos - num_defaults

    for idx, arg in enumerate(pos_args):
        if idx == len(args.posonlyargs) and args.posonlyargs:
            parts.append("/")
        text = arg.arg
        if arg.annotation is not None:
            text += f": {_unparse_safe(arg.annotation)}"
        default_idx = idx - default_offset
        if 0 <= default_idx < num_defaults:
            text += f" = {_unparse_safe(args.defaults[default_idx])}"
        parts.append(text)

    if args.posonlyargs and len(args.args) == 0:
        parts.append("/")

    if args.vararg is not None:
        vararg_text = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            vararg_text += f": {_unparse_safe(args.vararg.annotation)}"
        parts.append(vararg_text)
    elif args.kwonlyargs:
        parts.append("*")

    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        text = kw_arg.arg
        if kw_arg.annotation is not None:
            text += f": {_unparse_safe(kw_arg.annotation)}"
        if kw_default is not None:
            text += f" = {_unparse_safe(kw_default)}"
        parts.append(text)

    if args.kwarg is not None:
        kwarg_text = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            kwarg_text += f": {_unparse_safe(args.kwarg.annotation)}"
        parts.append(kwarg_text)

    return ", ".join(parts)


def _format_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    params = _format_arguments(node.args)
    ret = f" -> {_unparse_safe(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({params}){ret}"


def _format_class_signature(node: ast.ClassDef) -> str:
    bases = [_unparse_safe(b) for b in node.bases]
    keywords = [
        f"{kw.arg}={_unparse_safe(kw.value)}" if kw.arg else f"**{_unparse_safe(kw.value)}"
        for kw in node.keywords
    ]
    all_args = bases + keywords
    if all_args:
        return f"class {node.name}({', '.join(all_args)})"
    return f"class {node.name}"


def _node_span(node: ast.AST, total_lines: int) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", [])
    lineno = getattr(node, "lineno", 1)
    if decorators:
        start_line = min(lineno, *(getattr(d, "lineno", lineno) for d in decorators))
    else:
        start_line = lineno
    end_line = getattr(node, "end_lineno", None) or start_line
    start_line = max(1, min(start_line, total_lines or 1))
    end_line = max(start_line, min(end_line, total_lines or start_line))
    return start_line, end_line


def _slice_source(lines: list[str], start_line: int, end_line: int) -> str:
    if not lines:
        return ""
    return "\n".join(lines[start_line - 1 : end_line])


def _derive_module_path(file_path: Path, root_dir: Path | None = None) -> str:
    """Convert a Python file path into a dotted module path."""
    target = file_path
    if root_dir is not None:
        try:
            target = file_path.resolve().relative_to(root_dir.resolve())
        except ValueError:
            target = file_path
    parts = list(target.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else file_path.stem


class ManimASTChunker:
    """Parses ManimCE Python source files into structured AST CodeChunks."""

    def __init__(
        self,
        *,
        include_private: bool = False,
        include_module_chunk: bool = False,
        include_nested_functions: bool = False,
        on_syntax_error: Literal["skip", "raise"] = "skip",
    ) -> None:
        self.include_private = include_private
        self.include_module_chunk = include_module_chunk
        self.include_nested_functions = include_nested_functions
        self.on_syntax_error = on_syntax_error

    def _should_skip_name(self, name: str) -> bool:
        if self.include_private:
            return False
        if name.startswith("__") and name.endswith("__"):
            return False
        return name.startswith("_")

    def chunk_source(
        self,
        source: str,
        *,
        file_path: str = "<memory>",
        module_path: str = "manim_module",
    ) -> list[CodeChunk]:
        """Parse a Python source string into CodeChunks."""
        tree = ast.parse(source, filename=file_path)
        lines = source.splitlines()
        total_lines = len(lines)
        chunks: list[CodeChunk] = []
        seen_ids: dict[str, int] = {}

        def make_chunk_id(qualified_name: str, start_line: int) -> str:
            base = f"{module_path}:{qualified_name}" if module_path else qualified_name
            count = seen_ids.get(base, 0) + 1
            seen_ids[base] = count
            if count == 1:
                return base
            digest = hashlib.sha1(f"{base}:{start_line}:{count}".encode("utf-8")).hexdigest()[:8]
            return f"{base}#{digest}"

        if self.include_module_chunk:
            mod_doc = ast.get_docstring(tree) or ""
            chunks.append(
                CodeChunk(
                    chunk_id=make_chunk_id("__module__", 1),
                    chunk_type=ChunkType.MODULE,
                    name=module_path.split(".")[-1] if module_path else file_path,
                    qualified_name=module_path or file_path,
                    module_path=module_path,
                    file_path=file_path,
                    signature=f"module {module_path}",
                    docstring=mod_doc,
                    source=source,
                    start_line=1,
                    end_line=max(1, total_lines),
                    metadata={"is_module_summary": True},
                )
            )

        def visit_body(body: list[ast.stmt], class_stack: list[str]) -> None:
            for stmt in body:
                if isinstance(stmt, ast.ClassDef):
                    if self._should_skip_name(stmt.name):
                        continue
                    start_line, end_line = _node_span(stmt, total_lines)
                    parent_class = ".".join(class_stack) if class_stack else None
                    qualified_name = (
                        f"{'.'.join(class_stack)}.{stmt.name}" if class_stack else stmt.name
                    )
                    decorators = [_unparse_safe(d) for d in stmt.decorator_list]
                    bases = [_unparse_safe(b) for b in stmt.bases]
                    chunks.append(
                        CodeChunk(
                            chunk_id=make_chunk_id(qualified_name, start_line),
                            chunk_type=ChunkType.CLASS,
                            name=stmt.name,
                            qualified_name=qualified_name,
                            module_path=module_path,
                            file_path=file_path,
                            parent_class=parent_class,
                            signature=_format_class_signature(stmt),
                            decorators=decorators,
                            bases=bases,
                            docstring=ast.get_docstring(stmt) or "",
                            source=_slice_source(lines, start_line, end_line),
                            start_line=start_line,
                            end_line=end_line,
                            metadata={
                                "is_nested_class": bool(class_stack),
                                "method_count": sum(
                                    1
                                    for item in stmt.body
                                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                ),
                            },
                        )
                    )
                    visit_body(stmt.body, [*class_stack, stmt.name])

                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if self._should_skip_name(stmt.name):
                        continue
                    start_line, end_line = _node_span(stmt, total_lines)
                    parent_class = ".".join(class_stack) if class_stack else None
                    qualified_name = (
                        f"{'.'.join(class_stack)}.{stmt.name}" if class_stack else stmt.name
                    )
                    if class_stack:
                        chunk_type = ChunkType.METHOD
                    elif isinstance(stmt, ast.AsyncFunctionDef):
                        chunk_type = ChunkType.ASYNC_FUNCTION
                    else:
                        chunk_type = ChunkType.FUNCTION

                    decorators = [_unparse_safe(d) for d in stmt.decorator_list]
                    chunks.append(
                        CodeChunk(
                            chunk_id=make_chunk_id(qualified_name, start_line),
                            chunk_type=chunk_type,
                            name=stmt.name,
                            qualified_name=qualified_name,
                            module_path=module_path,
                            file_path=file_path,
                            parent_class=parent_class,
                            signature=_format_function_signature(stmt),
                            decorators=decorators,
                            docstring=ast.get_docstring(stmt) or "",
                            source=_slice_source(lines, start_line, end_line),
                            start_line=start_line,
                            end_line=end_line,
                            metadata={
                                "is_async": isinstance(stmt, ast.AsyncFunctionDef),
                                "arg_names": [
                                    a.arg
                                    for a in (
                                        list(stmt.args.posonlyargs)
                                        + list(stmt.args.args)
                                        + list(stmt.args.kwonlyargs)
                                    )
                                ],
                            },
                        )
                    )
                    if self.include_nested_functions:
                        visit_nested_functions(stmt.body, [*class_stack, stmt.name])

                elif isinstance(stmt, ast.If):
                    visit_body(stmt.body, class_stack)
                    visit_body(stmt.orelse, class_stack)
                elif isinstance(stmt, ast.Try):
                    visit_body(stmt.body, class_stack)
                    for handler in stmt.handlers:
                        visit_body(handler.body, class_stack)
                    visit_body(stmt.orelse, class_stack)
                    visit_body(stmt.finalbody, class_stack)
                elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                    visit_body(stmt.body, class_stack)
                elif hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar):
                    visit_body(stmt.body, class_stack)
                    for handler in stmt.handlers:
                        visit_body(handler.body, class_stack)
                    visit_body(stmt.orelse, class_stack)
                    visit_body(stmt.finalbody, class_stack)

        def visit_nested_functions(body: list[ast.stmt], fn_stack: list[str]) -> None:
            for stmt in body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if self._should_skip_name(stmt.name):
                        continue
                    start_line, end_line = _node_span(stmt, total_lines)
                    qualified_name = f"{'.'.join(fn_stack)}.{stmt.name}"
                    chunk_type = (
                        ChunkType.ASYNC_FUNCTION
                        if isinstance(stmt, ast.AsyncFunctionDef)
                        else ChunkType.FUNCTION
                    )
                    chunks.append(
                        CodeChunk(
                            chunk_id=make_chunk_id(qualified_name, start_line),
                            chunk_type=chunk_type,
                            name=stmt.name,
                            qualified_name=qualified_name,
                            module_path=module_path,
                            file_path=file_path,
                            parent_class=None,
                            signature=_format_function_signature(stmt),
                            decorators=[_unparse_safe(d) for d in stmt.decorator_list],
                            docstring=ast.get_docstring(stmt) or "",
                            source=_slice_source(lines, start_line, end_line),
                            start_line=start_line,
                            end_line=end_line,
                            metadata={"is_nested_function": True},
                        )
                    )
                    visit_nested_functions(stmt.body, [*fn_stack, stmt.name])

        visit_body(tree.body, [])
        return chunks

    def chunk_file(
        self,
        file_path: str | Path,
        *,
        root_dir: str | Path | None = None,
    ) -> ChunkingResult:
        """Parse a Python file on disk into structured CodeChunks."""
        path = Path(file_path)
        root = Path(root_dir) if root_dir is not None else None
        module_path = _derive_module_path(path, root)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            if self.on_syntax_error == "raise":
                raise
            return ChunkingResult(
                errors=[
                    ParseErrorInfo(
                        file_path=str(path),
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                ],
                files_processed=0,
            )

        try:
            chunks = self.chunk_source(
                source,
                file_path=str(path),
                module_path=module_path,
            )
            return ChunkingResult(chunks=chunks, files_processed=1)
        except SyntaxError as exc:
            if self.on_syntax_error == "raise":
                raise
            logger.warning("Skipping %s due to SyntaxError: %s", path, exc)
            return ChunkingResult(
                errors=[
                    ParseErrorInfo(
                        file_path=str(path),
                        error_type="SyntaxError",
                        message=exc.msg or str(exc),
                        line_number=exc.lineno,
                    )
                ],
                files_processed=0,
            )

    def chunk_directory(
        self,
        directory: str | Path,
        *,
        exclude_dirs: tuple[str, ...] = ("__pycache__", ".git", ".venv", "tests"),
    ) -> ChunkingResult:
        """Recursively parse all Python files under a directory."""
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")

        aggregated = ChunkingResult()
        py_files = sorted(
            p
            for p in root.rglob("*.py")
            if not any(part in exclude_dirs for part in p.relative_to(root).parts)
        )
        for py_file in py_files:
            res = self.chunk_file(py_file, root_dir=root)
            aggregated.chunks.extend(res.chunks)
            aggregated.errors.extend(res.errors)
            aggregated.files_processed += res.files_processed
        return aggregated
