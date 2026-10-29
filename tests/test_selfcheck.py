"""Self-check answer parsing (no model calls)."""

from __future__ import annotations

import pytest

from phantom import selfcheck
from phantom.http import cache_key


@pytest.mark.parametrize("reply,answer", [
    ("No.", "no"), ("yes", "yes"), ("**Yes**, it does.", "yes"), ("I'm not sure", "unclear"),
    ("", "unclear"), ("Nope", "unclear"), ("No, but `numpy` does.", "no"),
])
def test_parse_answer(reply, answer):
    assert selfcheck.parse_answer(reply) == answer


def test_score_is_one_only_for_no():
    assert [selfcheck.score(a) for a in ("no", "yes", "unclear")] == [1.0, 0.0, 0.0]


def test_fully_cached_answers_need_no_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr(selfcheck, "cache_dir", lambda model=selfcheck.MODEL: tmp_path)
    (tmp_path / f"{cache_key('zzq-ghostlib')}.json").write_text('{"name": "zzq-ghostlib", "answer": "no"}')

    def offline(*a, **k):
        raise AssertionError("Ollama must not be contacted when every answer is cached")

    monkeypatch.setattr(selfcheck.generate, "ollama_tags", offline)
    assert selfcheck.ask_all(["zzq-ghostlib"]) == {"zzq-ghostlib": {"name": "zzq-ghostlib", "answer": "no"}}
