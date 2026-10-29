"""Self-check answer parsing (no model calls)."""

from __future__ import annotations

import pytest

from phantom import selfcheck


@pytest.mark.parametrize("reply,answer", [
    ("No.", "no"), ("yes", "yes"), ("**Yes**, it does.", "yes"), ("I'm not sure", "unclear"),
    ("", "unclear"), ("Nope", "unclear"), ("No, but `numpy` does.", "no"),
])
def test_parse_answer(reply, answer):
    assert selfcheck.parse_answer(reply) == answer


def test_score_is_one_only_for_no():
    assert [selfcheck.score(a) for a in ("no", "yes", "unclear")] == [1.0, 0.0, 0.0]
