"""Evaluation helpers (BUILD_SPEC §8): folds, metrics, bootstrap CIs, calibration,
the guard threshold and the features that must be recomputed per fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

from phantom import config

N_FOLDS = 5
N_BOOTSTRAP = 1000
CALIBRATION_BINS = 10
TARGET_RECALL = 0.90
TARGET_PRECISION = 0.95


def assign_folds(groups: Sequence[str], n_splits: int = N_FOLDS, seed: int = config.GLOBAL_SEED) -> dict[str, int]:
    """Prompt id -> fold, from a shuffled, seeded ``GroupKFold`` (rows of a prompt stay together)."""
    groups = np.asarray(groups)
    folds: dict[str, int] = {}
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(groups, groups=groups)):
        folds.update({g: k for g in np.unique(groups[test])})
    return dict(sorted(folds.items()))


def precision_at_recall(y: np.ndarray, score: np.ndarray, target: float = TARGET_RECALL) -> float:
    """Best precision among thresholds whose recall is at least ``target``."""
    precision, recall, _ = precision_recall_curve(y, score)
    ok = recall >= target
    return float(precision[ok].max()) if ok.any() else float("nan")


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float = TARGET_PRECISION) -> float:
    """Best recall among thresholds whose precision is at least ``target`` (0 if none)."""
    precision, recall, _ = precision_recall_curve(y, score)
    ok = precision >= target
    return float(recall[ok].max()) if ok.any() else 0.0


def calibration_table(y: np.ndarray, prob: np.ndarray, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Equal-width reliability bins: count, mean predicted probability, observed ghost share."""
    idx = np.clip((prob * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "prob": prob, "y": y})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("prob", "mean"), frac_ghost=("y", "mean"))
    out = out.reindex(range(bins))
    out["n"] = out["n"].fillna(0).astype(int)
    out.insert(0, "lo", np.arange(bins) / bins)
    out.insert(1, "hi", (np.arange(bins) + 1) / bins)
    return out.reset_index(drop=True)


def ece(y: np.ndarray, prob: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    """Expected calibration error over equal-width bins."""
    t = calibration_table(y, prob, bins).dropna()
    return float((t["n"] * (t["mean_pred"] - t["frac_ghost"]).abs()).sum() / len(y))


def metrics(y: np.ndarray, score: np.ndarray, probabilistic: bool) -> dict[str, float]:
    """§8 metrics. Brier and ECE only make sense for scores in [0, 1] that mean a probability."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    return {
        "n": int(len(y)),
        "n_ghost": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)) if probabilistic else float("nan"),
        "precision_at_recall90": precision_at_recall(y, score),
        "recall_at_precision95": recall_at_precision(y, score),
        "ece": ece(y, score) if probabilistic else float("nan"),
    }


def _group_resamples(groups: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    """Row indices for ``n`` bootstrap resamples drawn by group (prompt)."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    return [np.concatenate([rows_by_group[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n)]


def bootstrap_auroc(y: np.ndarray, score: np.ndarray, groups: np.ndarray,
                    n: int = N_BOOTSTRAP, seed: int = config.GLOBAL_SEED) -> tuple[float, float]:
    """95% percentile CI for AUROC, resampling whole prompt groups."""
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    stats = [roc_auc_score(y[i], score[i]) for i in _group_resamples(np.asarray(groups), n, seed)
             if 0 < y[i].sum() < len(i)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired_auroc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray,
                      n: int = N_BOOTSTRAP, seed: int = config.GLOBAL_SEED) -> dict[str, float]:
    """AUROC(a) - AUROC(b) with a 95% CI from the same prompt-group resamples for both."""
    y = np.asarray(y).astype(int)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = np.array([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
                      for i in _group_resamples(np.asarray(groups), n, seed) if 0 < y[i].sum() < len(i)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "lo": float(lo), "hi": float(hi),
            "share_a_better": float((diffs > 0).mean())}


def bootstrap_rate(y: np.ndarray, groups: np.ndarray, n: int = N_BOOTSTRAP,
                   seed: int = config.GLOBAL_SEED) -> tuple[float, float, float]:
    """Share of positives with a 95% CI, resampling whole prompt groups."""
    y = np.asarray(y, dtype=float)
    stats = [y[i].mean() for i in _group_resamples(np.asarray(groups), n, seed)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(y.mean()), float(lo), float(hi)


def threshold_for_recall(y: np.ndarray, prob: np.ndarray, target: float = TARGET_RECALL) -> dict[str, Any]:
    """The highest tau (flag ghost if p >= tau) whose recall is still >= ``target``.

    "Smallest tau that reaches recall >= 0.90" (§8) read as the strictest threshold that
    keeps the recall target: any lower tau only adds false alarms.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    ghost_probs = np.sort(prob[y == 1])[::-1]
    k = int(np.ceil(target * len(ghost_probs)))  # ghosts we must catch
    tau = float(ghost_probs[k - 1])
    flagged = prob >= tau
    tp = int((flagged & (y == 1)).sum())
    return {"tau": tau, "target_recall": target, "recall": tp / int(y.sum()),
            "precision": tp / int(flagged.sum()), "n_flagged": int(flagged.sum()), "n": int(len(y))}


def with_cross_model(df: pd.DataFrame, known_models: set[str]) -> pd.DataFrame:
    """Recompute ``cross_model`` from the models a fold may know about.

    Each row counts the distinct models among ``known_models`` plus its own model that
    suggested the same name for the same prompt. With every model known this is the
    stored value; in leave-one-model-out the held-out model's mentions never reach the
    training rows, while its own rows still count themselves.
    """
    known = df[df["model"].isin(known_models)]
    seen = known.groupby(["prompt_id", "name"])["model"].agg(set)
    out = df.copy()
    out["cross_model"] = [len(seen.get((p, n), set()) | {m})
                          for p, n, m in zip(df["prompt_id"], df["name"], df["model"])]
    return out
