"""Feature tests: hand-checked name features (BUILD_SPEC §11), context features,
per-fold global_freq, logprob alignment. Uses a tiny fixed top list (no network)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from phantom import config, features
from phantom.charlm import CharTrigramLM
from phantom.topnames import TopIndex

TOP = TopIndex(["requests", "numpy", "six", "pip", "pdfminer", "opencv-python"])
LM = CharTrigramLM(TOP.names)


# name: (len, n_parts, has_digit, in_top15k, nn_lev, nearest rank, affix_of_real) -- by hand
HAND_CHECKED = {
    "requests": (8, 1, 0, 1, 0, 1, 0),
    "requests-utils-pro": (18, 3, 0, 0, 10, 1, 0),  # delete "-utils-pro"; two affixes != one
    "pdfminer-tables": (15, 2, 0, 0, 7, 5, 0),  # delete "-tables"
    "numpy2": (6, 1, 1, 0, 1, 2, 1),  # numpy + "2"
    "cv2": (3, 1, 1, 0, 3, 3, 0),  # 3 substitutions to six (rank 3) and pip (rank 4): tie -> six
}


@pytest.mark.parametrize("name", HAND_CHECKED)
def test_hand_checked_name_features(name):
    f = features.name_features(name, TOP, LM)
    length, parts, digit, top, lev, rank, affix = HAND_CHECKED[name]
    assert (f["len"], f["n_parts"], f["has_digit"], f["in_top15k"], f["nn_lev"], f["affix_of_real"]) == (
        length, parts, digit, top, lev, affix)
    assert f["nn_rank_log"] == pytest.approx(math.log10(rank))
    assert 0.0 <= f["nn_jw"] <= 1.0


def test_affix_both_directions():
    top = TopIndex(["pyyaml", "requests"])
    assert top.affix_match("yaml") == "pyyaml"  # real minus "py"
    assert top.affix_match("requests-api") == "requests"  # real plus "-api"
    assert top.affix_match("requests") is None


def test_char_lm_prefers_name_like_strings():
    lm = CharTrigramLM(["requests", "requests-toolbelt", "numpy", "pandas", "pytest", "pyyaml"])
    assert lm.mean_logprob("requests-oauth") > lm.mean_logprob("qxzvkj-wq")
    assert lm.zscore("pytest") > lm.zscore("qxzvkj-wq")


def labelled_frame():
    rows = [
        # key, model, prompt, name, raw, source, position, category
        ("a|p1|s1", "a", "p001", "requests", "requests", "pip_line", 0, "real"),
        ("a|p1|s2", "a", "p001", "requests", "requests", "pip_line", 1, "real"),
        ("a|p1|s1", "a", "p001", "ghostlib-zz", "ghostlib-zz", "pip_line", 1, "ghost"),
        ("b|p1|s1", "b", "p001", "ghostlib-zz", "ghostlib-zz", "pip_line", 0, "ghost"),
        ("b|p1|s1", "b", "p001", "json", "json", "pip_line", 2, "stdlib_confusion"),
    ]
    df = pd.DataFrame(rows, columns=["key", "model", "prompt_id", "name", "raw_token", "source",
                                     "position", "category"])
    df["y_ghost"] = df["category"] == "ghost"
    return df


def build_small():
    gens = pd.DataFrame({"key": ["a|p1|s1", "a|p1|s2", "b|p1|s1", "b|p1|s2"],
                         "model": ["a", "a", "b", "b"], "prompt_id": ["p001"] * 4})
    prompts = [{"id": "p001", "domain": "data-files", "niche_level": 3}]
    return features.build(labelled_frame(), gens, prompts, {}, TOP, LM, features.categories(["a", "b"]))


def test_context_features_and_subtypes_excluded():
    ds = build_small().set_index(["model", "name"])
    assert ("b", "json") not in ds.index  # subtype rows are not in the dataset
    assert ds.loc[("a", "requests"), "consistency"] == 1.0  # 2 of 2 samples
    assert ds.loc[("a", "ghostlib-zz"), "consistency"] == 0.5
    assert ds.loc[("a", "ghostlib-zz"), "cross_model"] == 2
    assert ds.loc[("a", "requests"), "position"] == 0
    assert ds.loc[("a", "requests"), "niche_level"] == 3
    assert math.isnan(ds.loc[("a", "requests"), "name_logprob"])  # no logprobs given


def test_feature_matrix_columns_and_no_registry_columns():
    X = features.feature_matrix(features.with_global_freq(build_small(), pd.Series(True, index=range(3))))
    assert list(X.columns) == features.FEATURES
    assert not set(features.FEATURES) & config.REGISTRY_COLUMNS


def test_features_do_not_depend_on_labels():
    ds = build_small()
    flipped = labelled_frame().assign(y_ghost=lambda d: ~d["y_ghost"])
    gens = pd.DataFrame({"key": ["a|p1|s1", "a|p1|s2", "b|p1|s1", "b|p1|s2"],
                         "model": ["a", "a", "b", "b"], "prompt_id": ["p001"] * 4})
    other = features.build(flipped, gens, [{"id": "p001", "domain": "data-files", "niche_level": 3}],
                           {}, TOP, LM, features.categories(["a", "b"]))
    cols = [c for c in features.FEATURES if c != "global_freq"]
    pd.testing.assert_frame_equal(
        ds.rename(columns={"model": "t", "model_code": "model"})[cols],
        other.rename(columns={"model": "t", "model_code": "model"})[cols])


def test_global_freq_uses_training_rows_only():
    df = pd.DataFrame({"name": ["x", "x", "x", "y"], "n_gens": [2, 1, 1, 1]})
    train = pd.Series([True, True, False, False])
    gf = features.with_global_freq(df, train)["global_freq"].tolist()
    # train rows see the train total (3); the test row sees train (3) + its own (1);
    # "y" is unseen in training, so only its own occurrence counts.
    assert gf == [3, 3, 4, 1]


def test_name_logprob_aligns_tokens():
    rec = {"logprobs": [["pip", -0.1], [" install", -0.2], [" ghost", -2.0], ["lib", -3.0], ["\n", -0.1]]}
    assert features.name_logprob(rec, "ghostlib") == pytest.approx(-2.5)
    assert math.isnan(features.name_logprob(rec, "absent"))
    assert math.isnan(features.name_logprob({"logprobs": None}, "ghostlib"))
