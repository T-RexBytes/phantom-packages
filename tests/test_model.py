"""Guard model wrapper: features, threshold file, batching and band (fake estimators, no TabPFN)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from phantom import features, model
from phantom.charlm import CharTrigramLM
from phantom.topnames import TopIndex

TOP = TopIndex(["requests", "numpy", "six", "pip", "pdfminer", "opencv-python"])


@pytest.fixture(autouse=True)
def tiny_index(monkeypatch):
    monkeypatch.setattr(model, "_index_and_lm", lambda: (TOP, CharTrigramLM(TOP.names)))


def test_name_feature_frame_columns_and_values():
    df = model.name_feature_frame(["requests", "zzq-ghostlib", "numpy2"])
    assert list(df.columns) == features.NAME_FEATURES == model.GUARD_FEATURES
    assert df["in_top15k"].tolist() == [1, 0, 0]
    assert df.loc[2, "affix_of_real"] == 1 and df.loc[0, "nn_lev"] == 0


def test_guard_tau_reads_threshold_file(tmp_path, monkeypatch):
    path = tmp_path / "guard_threshold.json"
    path.write_text(json.dumps({"tau": 0.42, "recall": 0.9}), encoding="utf-8")
    monkeypatch.setattr(model, "GUARD_THRESHOLD_PATH", path)
    assert model.guard_tau() == 0.42
    assert model.GuardModel().tau == 0.42


class FakeEstimator:
    """Returns p = offset + 0.01 * nn_lev; counts predict calls."""

    def __init__(self, offset: float) -> None:
        self.offset, self.calls = offset, 0

    def predict_proba(self, X):
        self.calls += 1
        lev = X[:, features.NAME_FEATURES.index("nn_lev")]
        p = np.clip(self.offset + 0.01 * lev, 0, 1)
        return np.column_stack([1 - p, p])


def fake_loader(monkeypatch):
    made = {}

    def load(path):
        est = FakeEstimator(0.5 if path == model.GUARD_MODEL_PATH else 0.1 * (model.GUARD_FOLD_PATHS.index(path) + 1))
        made[path] = est
        return est

    monkeypatch.setattr(model, "load_estimator", load)
    return made


def test_guard_model_is_lazy_and_batched(monkeypatch):
    made = fake_loader(monkeypatch)
    gm = model.GuardModel()
    assert not made  # nothing loaded at construction
    out = gm.predict(["requests", "zzq-ghostlib"])
    assert list(made) == [model.GUARD_MODEL_PATH] and made[model.GUARD_MODEL_PATH].calls == 1
    assert out["requests"] == (pytest.approx(0.5), None, None)
    assert out["zzq-ghostlib"][0] > out["requests"][0]
    assert gm.predict([]) == {}


def test_guard_model_band_spans_fold_models(monkeypatch):
    made = fake_loader(monkeypatch)
    out = model.GuardModel(band=True).predict(["requests"])
    p, lo, hi = out["requests"]
    assert len(made) == 6 and all(e.calls == 1 for e in made.values())
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.5) and p == pytest.approx(0.5)


def test_import_does_not_load_torch():
    assert "phantom.model" in sys.modules
    # model.py must import torch/tabpfn lazily (the hook imports it for every Bash call)
    src = model.__file__
    text = open(src, encoding="utf-8").read()
    top = text.split("\ndef ", 1)[0]
    assert "import torch" not in top and "import tabpfn" not in top and "from tabpfn" not in top
