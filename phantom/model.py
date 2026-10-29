"""The guard's TabPFN model: TabPFN-3.5-fast on name features only (BUILD_SPEC §9.1, deviation noted).

The guard only ever sees a package name, never the generation context, so it uses a model
trained on ``features.NAME_FEATURES``. Measured on the Phase 5 folds this is as accurate as the
spec's "context model with default context" and better calibrated (``results/summary.json``
→ ``guard_model``). torch/tabpfn are imported lazily so the hook stays fast when a command has
no installs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from phantom import config, features

GUARD_FEATURES: list[str] = features.NAME_FEATURES
GUARD_MODEL_PATH = config.MODEL_DIR / "guard_model.tabpfn_fit"
GUARD_FOLD_PATHS = [config.MODEL_DIR / f"guard_fold{k}.tabpfn_fit" for k in range(5)]
GUARD_THRESHOLD_PATH = config.MODEL_DIR / "guard_threshold.json"


def guard_tau() -> float:
    """The guard threshold (flag ghost if p >= tau), from out-of-fold probabilities."""
    return float(json.loads(GUARD_THRESHOLD_PATH.read_text(encoding="utf-8"))["tau"])


@lru_cache(maxsize=1)
def _index_and_lm():
    from phantom.charlm import CharTrigramLM
    from phantom.topnames import load_index

    index = load_index()
    return index, CharTrigramLM(index.names)


def name_feature_frame(names: list[str]) -> pd.DataFrame:
    """``GUARD_FEATURES`` for each name (one row per name, in order)."""
    index, lm = _index_and_lm()
    return pd.DataFrame([features.name_features(n, index, lm) for n in names], columns=GUARD_FEATURES)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_estimator(path) -> Any:
    """A fitted TabPFN estimator saved with ``save_fitted_tabpfn_model``."""
    config.load_dotenv()
    from tabpfn.model_loading import load_fitted_tabpfn_model

    return load_fitted_tabpfn_model(path, device=_device())


class GuardModel:
    """P(ghost) for package names; with ``band=True`` also the min–max over the 5 fold models."""

    def __init__(self, band: bool = False) -> None:
        self.band = band
        self._final: Any = None
        self._folds: list[Any] | None = None
        self._tau: float | None = None

    @property
    def tau(self) -> float:
        """Decision threshold from ``guard_threshold.json``."""
        if self._tau is None:
            self._tau = guard_tau()
        return self._tau

    def _load(self) -> None:
        if self._final is None:
            self._final = load_estimator(GUARD_MODEL_PATH)
        if self.band and self._folds is None:
            self._folds = [load_estimator(p) for p in GUARD_FOLD_PATHS]

    def predict(self, names: list[str]) -> dict[str, tuple[float, float | None, float | None]]:
        """name -> (p_ghost, band_lo, band_hi); one batched predict call per model."""
        if not names:
            return {}
        self._load()
        X = name_feature_frame(names).to_numpy(float)
        p = self._final.predict_proba(X)[:, 1]
        lo = hi = [None] * len(names)
        if self.band:
            fold_p = [m.predict_proba(X)[:, 1] for m in self._folds]
            lo = [float(min(fp[i] for fp in fold_p)) for i in range(len(names))]
            hi = [float(max(fp[i] for fp in fold_p)) for i in range(len(names))]
        return {n: (float(p[i]), lo[i], hi[i]) for i, n in enumerate(names)}
