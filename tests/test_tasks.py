"""Tests for the ``tasks.py`` stage runner."""

from __future__ import annotations

import importlib.util

import pytest

from phantom import config

spec = importlib.util.spec_from_file_location("tasks", config.ROOT / "tasks.py")
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


def test_stage_order_matches_pipeline():
    assert list(tasks.STAGES) == [
        "prompts", "summon", "extract", "features", "eval", "guard-model", "public-dataset", "eval-public", "squat", "replay", "ghost-types", "error-analysis", "figures", "halloween", "site",
    ]


def test_unknown_stage_is_usage_error():
    with pytest.raises(SystemExit) as exc:
        tasks.main(["nonsense"])
    assert exc.value.code == 2


def test_missing_stage_script_reports_phase(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tasks, "EXPERIMENTS", tmp_path)
    assert tasks.main(["eval"]) == 1
    assert "Phase 5" in capsys.readouterr().out
