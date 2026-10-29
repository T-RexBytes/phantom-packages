"""Stage 06 helpers on synthetic names (no network, no real ghost names)."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone

import pandas as pd

from phantom import config

spec = importlib.util.spec_from_file_location("squat_stage", config.ROOT / "experiments" / "06_squat_check.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


def candidates() -> pd.DataFrame:
    rows = [
        # name, exists_at_label, p, born_after, why, summons
        ("zz-still-missing", False, 0.9, False, "ghost_at_label", 3),
        ("zz-missing-once", False, 0.8, False, "ghost_at_label", 1),
        ("zz-new-body", False, 0.7, False, "ghost_at_label", 2),
        ("zz-real-young", True, 0.6, True, "born_after_summoning;p_ghost>=0.5", 2),
        ("zz-lookup-failed", False, 0.9, False, "ghost_at_label", 2),
    ]
    df = pd.DataFrame(rows, columns=["name", "exists_at_label", "p_ghost_oof", "born_after_summoning",
                                     "why_candidate", "summons"])
    return df.assign(first_release_at_label=None, mentions=df["summons"], n_models=1, n_prompts=1,
                     models="m1")


def registry_row(name: str, exists: bool, **kw) -> dict:
    return {"name": name, "exists": exists, "first_release": None, "last_release": None, "n_releases": None,
            "summary": None, "has_project_urls": None, "recent_downloads_month": None,
            "checked_at": "2026-10-30T00:00:00Z", **kw}


def assessed() -> pd.DataFrame:
    young = pd.Timestamp("2026-09-01", tz="UTC")
    now_rows = {
        "zz-still-missing": registry_row("zz-still-missing", False),
        "zz-missing-once": registry_row("zz-missing-once", False),
        "zz-new-body": registry_row("zz-new-body", True, first_release=young, n_releases=1,
                                    recent_downloads_month=12),
        "zz-real-young": registry_row("zz-real-young", True, first_release=young, n_releases=2,
                                      recent_downloads_month=40),
        "zz-lookup-failed": None,
    }
    return stage.assess(candidates(), now_rows, now=NOW)


def test_registered_during_study_only_for_label_time_ghosts_that_exist_now():
    df = assessed()
    by = df.set_index("name")
    assert by.loc["zz-new-body", "registered_during_study"] is True
    assert by.loc["zz-real-young", "registered_during_study"] is False  # existed at label time
    assert by.loc["zz-still-missing", "registered_during_study"] is False
    assert by.loc["zz-lookup-failed", "exists_now"] is None  # unknown, not "missing"


def test_squat_flags_follow_the_rules():
    df = assessed()
    by = df.set_index("name")
    assert by.loc["zz-new-body", "possible_slopsquat"] and by.loc["zz-real-young", "possible_slopsquat"]
    assert not by.loc["zz-still-missing", "possible_slopsquat"]


def test_public_frame_never_contains_unregistered_names():
    pub = stage.public_frame(assessed())
    assert set(pub["name"]) == {"zz-new-body", "zz-real-young"}
    assert pub["first_release"].tolist() == ["2026-09-01", "2026-09-01"]


def test_unregistered_repeated_and_anon_ids():
    df = assessed()
    rep = stage.unregistered_repeated(df)
    assert rep["name"].tolist() == ["zz-still-missing"]  # repeated, still 404; failed lookup excluded
    assert set(stage.still_unregistered_ghosts(df)["name"]) == {"zz-still-missing", "zz-missing-once"}
    assert rep["anon_id"].str.fullmatch(r"[0-9a-f]{8}").all()


def test_summary_counts_and_disclosure_draft():
    df = assessed()
    s = stage.summarize(df, failed=1, checked_at="2026-10-30T00:00:00Z")
    assert (s["n_candidates"], s["n_ghosts_at_label"], s["n_still_unregistered"]) == (5, 4, 2)
    assert (s["n_registered_during_study"], s["n_possible_slopsquat"], s["failed_lookups"]) == (1, 2, 1)
    assert s["n_unregistered_repeated"] == 1 and s["n_registered_candidates"] == 2
    text = stage.disclosure_markdown(df, s)
    assert text.startswith("# DRAFT: NOT SENT")
    assert "zz-new-body" in text and "zz-still-missing" not in text
    empty = stage.disclosure_markdown(df.iloc[0:0], s)
    assert "_none_" in empty
