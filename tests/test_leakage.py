"""Leakage guard: no registry-derived column may ever be a model feature (CLAUDE.md rule 7)."""

from __future__ import annotations

import json

import pytest

from phantom import config

FEATURE_LIST = config.MODEL_DIR / "feature_list.json"


def test_registry_columns_cover_spec_schema():
    # data/registry.parquet columns from BUILD_SPEC §4.4 plus the label.
    spec_cols = {
        "exists", "first_release", "last_release", "n_releases", "summary",
        "has_project_urls", "recent_downloads_month", "checked_at", "y_ghost",
    }
    assert spec_cols <= config.REGISTRY_COLUMNS


def test_feature_list_has_no_registry_columns():
    if not FEATURE_LIST.exists():
        pytest.skip("results/model/feature_list.json is built in Phase 4")
    features = set(json.loads(FEATURE_LIST.read_text(encoding="utf-8")))
    assert not features & config.REGISTRY_COLUMNS
