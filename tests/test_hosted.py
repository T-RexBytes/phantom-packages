"""Tests for Ultramarine cost accounting (no network)."""

from __future__ import annotations

import pytest

from phantom import hosted

# Price card shape as fetched from GET /v1/pricing on 2026-10-28.
CARD = {
    "uncached_input_nanodollars_per_token": 150,
    "cached_input_nanodollars_per_token": 30,
    "output_nanodollars_per_token": 500,
}


def test_cost_uses_nanodollar_rates():
    # (100-20)*150 + 20*30 + 300*500 = 12_000 + 600 + 150_000 nanodollars
    assert hosted.cost_usd(100, 20, 300, CARD) == pytest.approx(162_600 / 1e9)


def test_max_cost_bound_for_full_run_is_small():
    bound = hosted.max_cost_usd(200, 120, CARD)
    assert bound == pytest.approx(200 * (120 * 150 + hosted.MAX_COMPLETION_TOKENS * 500) / 1e9)
    assert bound < 0.15
