"""The public, name-free evaluation dataset (stage 12): shape, no names, no registry columns.
Checks the committed results/public/dataset.csv (CI-safe) plus the builder on synthetic rows."""

from __future__ import annotations

import importlib.util
import json
import re

import pandas as pd
import pytest

from phantom import config, features, privacy

_spec = importlib.util.spec_from_file_location("public12", config.ROOT / "experiments" / "12_public_dataset.py")
stage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage)


@pytest.fixture(scope="module")
def pub() -> pd.DataFrame:
    if not stage.DATASET_CSV.exists():
        pytest.skip("results/public/dataset.csv not built yet (experiments/12_public_dataset.py)")
    return pd.read_csv(stage.DATASET_CSV, dtype={"name_id": str})


def test_columns_are_ids_plus_features_minus_global_freq(pub):
    assert list(pub.columns) == stage.COLUMNS
    feats = {f for f in features.FEATURES if f != "global_freq"} - {"model"} | {"model_code"}
    assert set(stage.FEATURE_COLUMNS) == feats
    assert not {"name", "raw_token"} & set(pub.columns)
    assert not set(pub.columns) & (config.REGISTRY_COLUMNS - {"y_ghost"})


def test_no_name_can_leak(pub):
    assert pub["name_id"].astype(str).str.fullmatch(r"[0-9a-f]{8}").all()
    text_cols = [c for c in pub.columns if pd.api.types.is_string_dtype(pub[c])]
    assert set(text_cols) == {"name_id", "model", "prompt_id"}
    assert pub["prompt_id"].str.fullmatch(r"p\d{3}").all()


def test_counts_match_the_summary(pub):
    s = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert len(pub) == s["dataset"]["n_rows"] and int(pub["y_ghost"].sum()) == s["dataset"]["n_ghost"]
    assert pub["name_id"].nunique() == s["dataset"]["n_unique_names"]
    if privacy.ANON_IDS_PATH.exists():  # every ghost row's id is a published unregistered-ghost id
        assert set(pub.loc[pub["y_ghost"] == 1, "name_id"].astype(str)) <= privacy.load_anon_ids()


def test_public_frame_replaces_names_with_keyed_ids():
    ds = pd.DataFrame({"name": ["zzq-ghostlib", "requests"], "model": ["m", "m"], "prompt_id": ["p001", "p002"],
                       "y_ghost": [True, False], "n_gens": [1, 2], "model_code": [0, 0],
                       **{f: [0, 0] for f in stage.FEATURE_COLUMNS if f != "model_code"}})
    out = stage.public_frame(ds, {"p001": 0, "p002": 1}, key=b"test-key-not-a-secret")
    assert list(out.columns) == stage.COLUMNS and "name" not in out.columns
    assert out["name_id"].tolist() == [privacy.anon_id(n, b"test-key-not-a-secret") for n in ds["name"]]
    assert out["fold"].tolist() == [0, 1] and out["y_ghost"].tolist() == [1, 0]
    assert all(re.fullmatch(r"[0-9a-f]{8}", i) for i in out["name_id"])
