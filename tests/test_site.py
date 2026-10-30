"""The Ghost Registry site (BUILD_SPEC §5.8, §10): public-safe data and accessibility basics.

Skips the data checks until stage 08 has written web/data.json.
"""

from __future__ import annotations

import json
import re

import pytest

from phantom import config, privacy

WEB = config.ROOT / "web"
DATA = WEB / "data.json"
HTML = WEB / "index.html"


@pytest.fixture(scope="module")
def data():
    if not DATA.exists():
        pytest.skip("web/data.json not built yet (uv run tasks.py site)")
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_tombstones_are_anonymised(data):
    allowed = {"id", "domain", "niche_level", "mentions", "models", "n_prompts"}
    ids = privacy.load_anon_ids() if privacy.ANON_IDS_PATH.exists() else None
    for t in data["tombstones"]:
        assert set(t) == allowed  # no name field, ever
        assert re.fullmatch(r"[0-9a-f]{8}", t["id"])
        if ids is not None:
            assert t["id"] in ids  # every stone is a currently-unregistered ghost
    assert len(data["tombstones"]) == data["headline"]["n_ghost_names"] - data["headline"]["n_registered_during_study"]


def test_site_numbers_match_summary(data):
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert data["headline"]["ghost_rate"] == summary["rates"]["overall"]["ghost_rate"]
    tabpfn = next(r for r in data["eval"]["A"] if r["arm"] == "tabpfn")
    assert tabpfn["auroc"] == summary["eval"]["A"]["tabpfn"]["auroc"]
    assert {r["arm"] for r in data["eval"]["A"]} == set(summary["eval"]["A"])
    assert data["threshold"]["tau"] == summary["threshold"]["tau"]


def test_html_accessibility_basics():
    html = HTML.read_text(encoding="utf-8")
    imgs = re.findall(r"<img\b[^>]*>", html)
    assert imgs and all(re.search(r'\balt="[^"]{20,}"', i) for i in imgs)
    assert '<html lang="en">' in html and 'name="viewport"' in html
    assert "prefers-reduced-motion: reduce" in html and "max-width: 480px" in html
    assert not re.search(r"<script[^>]+src=", html)  # no third-party scripts
    for path in re.findall(r'src="(assets/[^"]+)"', html):
        assert (WEB / path).exists(), path
