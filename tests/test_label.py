"""Tests for mention building, labelling categories and rate arithmetic."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from phantom.label import build_mentions, label, rates

RECORDS = [
    {"key": "gemma4:e4b|p001|s1", "model": "gemma4:e4b", "prompt_id": "p001", "sample": 1,
     "output": "```bash\npip install requests fakepkg-zz9 cv2 json\n```\n```python\nimport yaml\nimport numpy\n```"},
    {"key": "phi4-mini:3.8b|p001|s1", "model": "phi4-mini:3.8b", "prompt_id": "p001", "sample": 1,
     "output": "```bash\npip install requests <pkg>\n```"},
]
REGISTRY = pd.DataFrame([
    {"name": "requests", "exists": True, "first_release": datetime(2011, 2, 14, tzinfo=timezone.utc)},
    {"name": "fakepkg-zz9", "exists": False, "first_release": None},
    {"name": "cv2", "exists": False, "first_release": None},
    {"name": "json", "exists": True, "first_release": datetime(2026, 9, 1, tzinfo=timezone.utc)},
    {"name": "pyyaml", "exists": True, "first_release": datetime(2006, 1, 1, tzinfo=timezone.utc)},
])


@pytest.fixture
def mentions():
    return build_mentions(RECORDS)


def test_build_mentions_sources_subtypes_and_drops(mentions):
    df, dropped = mentions
    assert dropped == 1  # "<pkg>"
    got = dict(zip(df["name"], df["subtype"]))
    assert got["cv2"] == "module_name_confusion" and got["json"] == "stdlib_confusion"
    assert pd.isna(got["requests"]) and pd.isna(got["pyyaml"])
    assert df.loc[df["name"] == "numpy", "source"].item() == "import_unmapped"


def test_label_categories_exclude_unmapped(mentions):
    lab = label(mentions[0], REGISTRY)
    cats = dict(zip(lab["key"] + "/" + lab["name"], lab["category"]))
    assert cats["gemma4:e4b|p001|s1/fakepkg-zz9"] == "ghost"
    assert cats["gemma4:e4b|p001|s1/cv2"] == "module_name_confusion"
    assert cats["gemma4:e4b|p001|s1/json"] == "stdlib_confusion"
    assert cats["gemma4:e4b|p001|s1/pyyaml"] == "real"
    assert "numpy" not in set(lab["name"])
    assert lab["y_ghost"].sum() == 1


def test_born_after_summoning_uses_model_release_date(mentions):
    lab = label(mentions[0], REGISTRY)
    # json's first release (2026-09) is after gemma4:e4b's 2026-03-02 date
    assert lab.loc[lab["name"] == "json", "born_after_summoning"].item()
    assert not lab.loc[lab["name"] == "requests", "born_after_summoning"].any()


def test_truncated_last_line_is_ignored():
    rec = {"key": "m|p1|s1", "model": "m", "prompt_id": "p1", "sample": 1, "done_reason": "length",
           "output": "```bash\npip install requests\npip install opencv-python-head"}
    df, _ = build_mentions([rec])
    assert list(df["name"]) == ["requests"]
    # a reply that finished normally keeps its last line
    df, _ = build_mentions([{**rec, "done_reason": "stop"}])
    assert "opencv-python-head" in set(df["name"])


def test_rates(mentions):
    lab = label(mentions[0], REGISTRY)
    gens = pd.DataFrame(RECORDS)[["key", "model"]]
    r = rates(lab, gens)
    assert r.loc["gemma4:e4b", "mentions"] == 3  # requests, fakepkg, pyyaml (subtypes excluded)
    assert r.loc["gemma4:e4b", "ghost_rate"] == pytest.approx(1 / 3)
    assert r.loc["phi4-mini:3.8b", "ghosts"] == 0
    assert r.loc["ALL", "gen_ghost_share"] == pytest.approx(1 / 2)
