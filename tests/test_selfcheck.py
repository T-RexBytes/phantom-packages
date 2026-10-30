"""Self-check answer parsing (no model calls)."""

from __future__ import annotations

import math

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
    monkeypatch.setattr(selfcheck, "cache_dir", lambda model=selfcheck.MODEL, **kw: tmp_path)
    (tmp_path / f"{cache_key('zzq-ghostlib')}.json").write_text('{"name": "zzq-ghostlib", "answer": "no"}')

    def offline(*a, **k):
        raise AssertionError("Ollama must not be contacted when every answer is cached")

    monkeypatch.setattr(selfcheck.generate, "ollama_tags", offline)
    assert selfcheck.ask_all(["zzq-ghostlib"]) == {"zzq-ghostlib": {"name": "zzq-ghostlib", "answer": "no"}}


def _pos(token, alts):
    """One logprob position as stored: [token, logprob, [[alt, logprob], ...]]."""
    return [token, alts[0][1], alts]


@pytest.mark.parametrize("positions,expected", [
    # both answers present: P(no) / (P(no) + P(yes)); case and spaces folded together
    ([_pos("No", [["No", -0.2], ["no", -1.5], [" Yes", -2.0], ["yes", -3.0], ["I", -5.0]])],
     (math.exp(-0.2) + math.exp(-1.5)) / (math.exp(-0.2) + math.exp(-1.5) + math.exp(-2.0) + math.exp(-3.0))),
    # only "no" variants in the top-k: yes has (near) zero mass
    ([_pos("no", [["no", -0.16], ["No", -1.9], ["I", -8.3]])], 1.0),
    ([_pos("Yes", [["Yes", -0.18], ["yes", -1.8], ["I", -15.0]])], 0.0),
    # leading whitespace/markup tokens are skipped; the first real token decides
    ([_pos("\n", [["\n", -0.01]]), _pos("**", [["**", -0.1]]), _pos("No", [["No", -0.3], ["Yes", -1.4]])],
     math.exp(-0.3) / (math.exp(-0.3) + math.exp(-1.4))),
])
def test_p_no_from_top_logprobs(positions, expected):
    assert selfcheck.p_no(positions) == pytest.approx(expected)


def test_p_no_is_none_when_neither_answer_is_in_the_top_k():
    assert selfcheck.p_no([_pos("I", [["I", -0.5], ["Based", -1.0], ["The", -2.0]])]) is None
    assert selfcheck.p_no([]) is None and selfcheck.p_no(None) is None


def test_prob_score_falls_back_to_the_hard_answer():
    assert selfcheck.prob_score({"answer": "no", "p_no": None}) == 1.0
    assert selfcheck.prob_score({"answer": "yes", "p_no": None}) == 0.0
    assert selfcheck.prob_score({"answer": "no", "p_no": 0.7}) == pytest.approx(0.7)
