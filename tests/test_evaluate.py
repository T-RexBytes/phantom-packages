"""Evaluation helpers (BUILD_SPEC §8): folds, metrics, bootstrap, threshold, per-fold features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phantom import evaluate


def test_folds_keep_prompts_together_and_are_deterministic():
    groups = [f"p{i % 23:03d}" for i in range(200)]
    a = evaluate.assign_folds(groups)
    assert a == evaluate.assign_folds(groups)
    assert set(a) == set(groups) and set(a.values()) == set(range(5))


def test_precision_at_recall_and_recall_at_precision():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .2, .6, .1, .1, .1, .1, .1])
    # recall >= 0.9 needs all 4 ghosts: threshold .2 flags 4 ghosts + 1 real (.6) -> 4/5
    assert evaluate.precision_at_recall(y, s, 0.9) == pytest.approx(0.8)
    # precision >= 0.95: only the top 3 (all ghosts) -> recall 3/4
    assert evaluate.recall_at_precision(y, s, 0.95) == pytest.approx(0.75)


def test_ece_perfect_and_bad():
    y = np.array([0] * 50 + [1] * 50)
    assert evaluate.ece(y, y.astype(float)) == pytest.approx(0.0)
    assert evaluate.ece(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_threshold_is_the_highest_tau_keeping_recall():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([.9, .8, .7, .3, .5, .1])
    t = evaluate.threshold_for_recall(y, p, 0.75)  # 3 of 4 ghosts suffice
    assert t["tau"] == pytest.approx(0.7) and t["recall"] == pytest.approx(0.75)
    assert t["precision"] == pytest.approx(1.0)
    t = evaluate.threshold_for_recall(y, p, 0.9)  # needs the .3 ghost -> also flags .5
    assert t["tau"] == pytest.approx(0.3) and t["precision"] == pytest.approx(0.8)


def test_bootstrap_ci_is_seeded_and_brackets_the_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 0.8, 300)
    g = np.repeat(np.arange(60), 5)
    lo, hi = evaluate.bootstrap_auroc(y, s, g, n=200, seed=1)
    assert (lo, hi) == evaluate.bootstrap_auroc(y, s, g, n=200, seed=1)
    from sklearn.metrics import roc_auc_score
    assert lo < roc_auc_score(y, s) < hi


def test_paired_diff_is_zero_for_identical_scores_and_positive_for_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good, noisy = y + rng.normal(0, 0.5, 300), y + rng.normal(0, 2.0, 300)
    g = np.repeat(np.arange(60), 5)
    same = evaluate.paired_auroc_diff(y, good, good, g, n=100)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    d = evaluate.paired_auroc_diff(y, good, noisy, g, n=200)
    assert d["lo"] > 0 and d["share_a_better"] == 1.0


def test_bootstrap_rate_ci():
    df = pd.DataFrame({"prompt_id": list("aabbccdd"), "y": [1, 0, 0, 0, 1, 1, 0, 0]})
    point, lo, hi = evaluate.bootstrap_rate(df["y"].to_numpy(), df["prompt_id"].to_numpy(), n=200, seed=0)
    assert point == pytest.approx(3 / 8) and 0 <= lo <= point <= hi <= 1


def test_cross_model_counts_only_known_models():
    df = pd.DataFrame({"model": ["a", "b", "c", "a"], "prompt_id": ["p1", "p1", "p1", "p2"],
                       "name": ["x", "x", "x", "x"], "cross_model": [3, 3, 3, 1]})
    out = evaluate.with_cross_model(df, {"a", "b"})["cross_model"].tolist()
    # a and b see each other (2); c is held out: training models (2) + itself = 3; p2 alone = 1
    assert out == [2, 2, 3, 1]
    assert evaluate.with_cross_model(df, {"a", "b", "c"})["cross_model"].tolist() == [3, 3, 3, 1]


def test_metrics_skip_calibration_for_raw_scores():
    y = np.array([0, 1, 0, 1])
    m = evaluate.metrics(y, np.array([1.0, 5.0, 2.0, 4.0]), probabilistic=False)
    assert m["auroc"] == 1.0 and np.isnan(m["brier"]) and np.isnan(m["ece"])
    m = evaluate.metrics(y, np.array([.1, .9, .2, .8]), probabilistic=True)
    assert m["brier"] == pytest.approx((.01 + .01 + .04 + .04) / 4)
