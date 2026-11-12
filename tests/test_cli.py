"""Tests for the manim-rag CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manim_rag.cli import main

SAMPLE_SOURCE_DIR = Path(__file__).resolve().parent.parent / "examples" / "sample_manim_source"


def test_cli_index_search_and_dry_run_generate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kb_path = tmp_path / "kb.json"

    rc_idx = main(["index", str(SAMPLE_SOURCE_DIR), "-o", str(kb_path)])
    assert rc_idx == 0
    idx_out = json.loads(capsys.readouterr().out)
    assert idx_out["files_processed"] == 3
    assert idx_out["chunks_indexed"] > 0
    assert kb_path.exists()

    rc_search = main(
        ["search", "Axes plot curve", "--kb", str(kb_path), "--fusion", "rrf", "--top-k", "3"]
    )
    assert rc_search == 0
    search_out = json.loads(capsys.readouterr().out)
    assert len(search_out) == 3
    assert any("Axes" in item["qualified_name"] for item in search_out)

    rc_gen = main(
        ["generate", "Draw a circle with Create", "--kb", str(kb_path), "--dry-run"]
    )
    assert rc_gen == 0
    gen_out = json.loads(capsys.readouterr().out)
    assert gen_out["mode"] == "dry-run"
    assert len(gen_out["retrieved"]) > 0
