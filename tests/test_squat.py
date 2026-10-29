"""Squat rules and candidate selection (BUILD_SPEC §5.6). Synthetic names only."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from phantom import squat

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
FRESH = {"exists": True, "n_releases": 1, "first_release": datetime(2026, 5, 14, tzinfo=timezone.utc),
         "recent_downloads_month": 40}


def test_all_checks_true_is_possible_slopsquat():
    c = squat.checks(FRESH, summons=2, now=NOW)
    assert c == dict.fromkeys(squat.CHECKS, True)
    assert squat.possible_slopsquat(c)


@pytest.mark.parametrize("field,ok,bad", [
    ("n_releases", 3, 4),
    ("recent_downloads_month", 499, 500),
])
def test_boundaries(field, ok, bad):
    assert squat.possible_slopsquat(squat.checks({**FRESH, field: ok}, 2, NOW))
    assert not squat.possible_slopsquat(squat.checks({**FRESH, field: bad}, 2, NOW))


def test_age_window_edge():
    edge = NOW - timedelta(days=squat.DAYS_PER_MONTH * 24)
    assert squat.checks({**FRESH, "first_release": edge}, 2, NOW)["young"]
    assert not squat.checks({**FRESH, "first_release": edge - timedelta(days=1)}, 2, NOW)["young"]


def test_summons_boundary():
    assert not squat.checks(FRESH, 1, NOW)["repeated"]
    assert squat.checks(FRESH, 2, NOW)["repeated"]


@pytest.mark.parametrize("missing", [None, math.nan, pd.NA])
def test_missing_downloads_count_as_low(missing):
    assert squat.checks({**FRESH, "recent_downloads_month": missing}, 2, NOW)["low_downloads"]


def test_nat_first_release_and_pandas_types():
    row = {**FRESH, "first_release": pd.NaT, "n_releases": 2.0}
    c = squat.checks(row, 3, NOW)
    assert not c["young"] and c["few_releases"] and not squat.possible_slopsquat(c)
    ts = {**FRESH, "first_release": pd.Timestamp("2026-05-14T00:00:00Z")}
    assert squat.possible_slopsquat(squat.checks(ts, 2, NOW))


def test_missing_name_fails_registry_checks():
    ghost = {"exists": False, "n_releases": None, "first_release": None, "recent_downloads_month": None}
    c = squat.checks(ghost, 5, NOW)
    assert c == {"exists": False, "few_releases": False, "young": False, "low_downloads": False, "repeated": True}
    assert not squat.possible_slopsquat(c)
    assert squat.reasons(ghost, c, 5, NOW) == ["404 on PyPI", "summoned by 5 (model, sample) pairs"]


def test_reasons_are_factual_strings():
    c = squat.checks(FRESH, 3, NOW)
    assert squat.reasons(FRESH, c, 3, NOW) == [
        "1 release", "first release 2026-05-14 (5 months ago)", "~40 downloads/month",
        "summoned by 3 (model, sample) pairs"]
    no_dl = {**FRESH, "recent_downloads_month": None}
    assert "no download data" in squat.reasons(no_dl, squat.checks(no_dl, 1, NOW), 1, NOW)


def labelled():
    rows = [
        # key, model, prompt, name, category, born_after_summoning
        ("a|p001|s1", "a", "p001", "ghostlib-zz", "ghost", False),
        ("a|p002|s1", "a", "p002", "ghostlib-zz", "ghost", False),  # same (a, s1): counts once
        ("a|p001|s2", "a", "p001", "ghostlib-zz", "ghost", False),
        ("b|p001|s1", "b", "p001", "ghostlib-zz", "ghost", False),
        ("a|p003|s1", "a", "p003", "fresh-pkg", "real", True),
        ("b|p003|s2", "b", "p003", "fresh-pkg", "real", False),
        ("a|p004|s1", "a", "p004", "risky-real", "real", False),
        ("a|p005|s1", "a", "p005", "old-real", "real", False),
        ("a|p006|s1", "a", "p006", "json", "stdlib_confusion", False),
        ("a|p007|s1", "a", "p007", "both-pkg", "real", True),
    ]
    df = pd.DataFrame(rows, columns=["key", "model", "prompt_id", "name", "category", "born_after_summoning"])
    df["y_ghost"] = df["category"] == "ghost"
    return df


def test_summon_counts_distinct_model_sample_pairs():
    s = squat.summon_counts(labelled()).set_index("name")
    assert s.loc["ghostlib-zz", "summons"] == 3  # (a,s1) (a,s2) (b,s1)
    assert s.loc["ghostlib-zz", "mentions"] == 4
    assert s.loc["ghostlib-zz", "n_models"] == 2 and s.loc["ghostlib-zz", "n_prompts"] == 2
    assert s.loc["ghostlib-zz", "models"] == "a,b"
    assert "json" not in s.index  # subtypes are excluded


def test_select_candidates_reasons():
    reg = pd.DataFrame({
        "name": ["ghostlib-zz", "fresh-pkg", "risky-real", "old-real", "json", "both-pkg"],
        "exists": [False, True, True, True, False, True],
        "first_release": [pd.NaT, pd.Timestamp("2026-05-14", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC"),
                          pd.Timestamp("2010-01-01", tz="UTC"), pd.NaT, pd.Timestamp("2026-06-01", tz="UTC")],
    })
    oof = pd.DataFrame({"name": ["ghostlib-zz", "risky-real", "risky-real", "old-real", "both-pkg"],
                        "y_ghost": [1, 0, 0, 0, 0], "A_tabpfn": [0.9, 0.2, 0.7, 0.1, 0.8]})
    out = squat.select_candidates(labelled(), reg, oof).set_index("name")
    assert out.loc["ghostlib-zz", "why_candidate"] == "ghost_at_label"
    assert out.loc["fresh-pkg", "why_candidate"] == "born_after_summoning"
    assert out.loc["risky-real", "why_candidate"] == "p_ghost>=0.5"
    assert out.loc["risky-real", "p_ghost_oof"] == pytest.approx(0.7)  # max over rows
    assert out.loc["both-pkg", "why_candidate"] == "born_after_summoning;p_ghost>=0.5"
    assert math.isnan(out.loc["fresh-pkg", "p_ghost_oof"])
    assert "old-real" not in out.index and "json" not in out.index
    assert list(out.reset_index().columns) == [
        "name", "exists_at_label", "first_release_at_label", "p_ghost_oof", "born_after_summoning",
        "why_candidate", "summons", "mentions", "n_models", "n_prompts", "models"]
