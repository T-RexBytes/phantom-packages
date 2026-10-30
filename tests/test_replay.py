"""Replay verdict mapping (stage 09) on synthetic names: mirrors phantom.guard."""

from __future__ import annotations

import importlib.util
import math

import pandas as pd

from phantom import config

spec = importlib.util.spec_from_file_location("replay", config.ROOT / "experiments" / "09_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

CHECKED = "2026-10-29T18:00:00Z"
OLD = {"exists": True, "n_releases": 40, "first_release": pd.Timestamp("2015-01-01", tz="UTC"),
       "recent_downloads_month": 10_000_000, "checked_at": CHECKED}
FRESH = {"exists": True, "n_releases": 1, "first_release": pd.Timestamp("2026-09-01", tz="UTC"),
         "recent_downloads_month": 12, "checked_at": CHECKED}
YOUNG_BUSY = {**FRESH, "n_releases": 9, "recent_downloads_month": 50_000}


def test_online_verdicts_mirror_the_guard():
    assert replay.online_verdict("json", None, 0.9, 0.5) == ("OK", "standard library")
    assert replay.online_verdict("zzq-ghostlib", None, math.nan, 0.5)[0] == "GHOST"
    assert replay.online_verdict("zzq-ghostlib", {"exists": False}, 0.1, 0.5)[0] == "GHOST"
    assert replay.online_verdict("zzq-old", OLD, 0.99, 0.5)[0] == "OK"  # old: model alone never blocks
    assert replay.online_verdict("zzq-fresh", FRESH, 0.0, 0.5)[0] == "SUSPECT"  # squat rule
    assert replay.online_verdict("zzq-busy", YOUNG_BUSY, 0.9, 0.5)[0] == "SUSPECT"  # young + ghost-like
    assert replay.online_verdict("zzq-busy", YOUNG_BUSY, 0.2, 0.5)[0] == "OK"
    assert replay.online_verdict("zzq-busy", YOUNG_BUSY, math.nan, 0.5)[0] == "OK"


def test_offline_verdicts():
    assert replay.offline_verdict("json", 0.99, 0.5) == "OK"
    assert replay.offline_verdict("zzq-ghostlib", 0.6, 0.5) == "GHOST_LIKELY"
    assert replay.offline_verdict("zzq-ghostlib", 0.4, 0.5) == "OK_UNVERIFIED"
    assert replay.offline_verdict("zzq-ghostlib", math.nan, 0.5) == "OK_UNVERIFIED"


def test_summarize_counts_generations_and_false_blocks():
    m = pd.DataFrame({
        "key": ["g1", "g1", "g2", "g3"],
        "model": ["a", "a", "a", "a"],
        "name": ["zzq-ghostlib", "requests", "zzq-fresh", "json"],
        "category": ["ghost", "real", "real", "stdlib_confusion"],
        "online": ["GHOST", "OK", "SUSPECT", "OK"],
        "offline": ["GHOST_LIKELY", "OK_UNVERIFIED", "OK_UNVERIFIED", "OK"],
    })
    gens = pd.DataFrame({"key": ["g1", "g2", "g3", "g4"], "model": ["a"] * 4})
    s = replay.summarize(m, gens)
    assert s["generations"] == 4 and s["generations_with_install"] == 3
    assert s["online"]["generations_blocked"] == 2 and s["online"]["ghosts_blocked"] == 1
    assert s["online"]["real_blocked"] == 1 and s["online"]["real_blocked_share"] == 0.5
    assert s["offline"]["ghosts_caught"] == 1 and s["offline"]["real_flagged"] == 0
    assert s["subtype_mentions"]["stdlib_confusion"] == {"OK": 1}
