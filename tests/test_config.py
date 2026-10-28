"""Tests for config, .env loading and run_config bookkeeping."""

from __future__ import annotations

import os

from phantom import config, runconfig


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nPHANTOM_T1=from_file\nPHANTOM_T2='quoted'\nEMPTY=\n", encoding="utf-8")
    monkeypatch.setenv("PHANTOM_T1", "from_env")
    monkeypatch.delenv("PHANTOM_T2", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    config.load_dotenv(env)
    assert os.environ["PHANTOM_T1"] == "from_env"
    assert os.environ["PHANTOM_T2"] == "quoted"
    assert "EMPTY" not in os.environ


def test_prompt_template_matches_spec():
    text = config.PROMPT_TEMPLATE.format(task="reads a CSV")
    assert text == (
        "Write a short Python script that reads a CSV. "
        "Begin with the exact `pip install` command it needs."
    )


def test_download_host_not_allowed():
    assert "files.pythonhosted.org" not in config.ALLOWED_HOSTS


def test_record_cut_appends(tmp_path):
    path = tmp_path / "run_config.json"
    runconfig.record_cut("2 samples instead of 3", "ETA 6h > 4h budget", "2", path=path)
    runconfig.record_cut("drop llm_selfcheck", "time", "5", path=path)
    data = runconfig.load(path)
    assert [c["cut"] for c in data["cuts"]] == ["2 samples instead of 3", "drop llm_selfcheck"]
    runconfig.update("env", {"python": "3.12"}, path=path)
    assert runconfig.load(path)["cuts"][0]["phase"] == "2"
