"""Rerun the detector evaluation from the public, name-free dataset (results/public/dataset.csv).

Split A (GroupKFold by prompt, the committed folds) and split C (rows whose name_id never appears
in their training fold) for TabPFN-3.5 and gradient boosting, with exactly the settings and
per-fold features of experiments/05_eval.py. Prints the metrics next to the committed
results/eval.csv values. Needs only the repo (plus TABPFN_TOKEN for the first weight download).
"""

from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pandas as pd

from phantom import config, evaluate, features

PUBLIC_CSV = config.RESULTS / "public" / "dataset.csv"
ARMS = ["hgb", "tabpfn"]
METRICS = ["auroc", "auprc", "precision_at_recall90", "brier"]

_spec = importlib.util.spec_from_file_location("eval05", config.ROOT / "experiments" / "05_eval.py")
eval05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval05)


def load_public(path=PUBLIC_CSV) -> pd.DataFrame:
    """Public table shaped like data/dataset.parquet (name_id plays the role of name)."""
    return pd.read_csv(path, dtype={"name_id": str}).rename(columns={"name_id": "name"})  # keep leading zeros


def run(ds: pd.DataFrame, arms=ARMS) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Out-of-fold scores per arm on split A, and the split-C (unseen name) mask."""
    y = ds["y_ghost"].to_numpy(int)
    oof = {arm: np.full(len(ds), np.nan) for arm in arms}
    unseen = np.zeros(len(ds), bool)
    for k in sorted(ds["fold"].unique()):
        test = (ds["fold"] == k).to_numpy()
        train = ~test
        frame = features.with_global_freq(ds, pd.Series(train, index=ds.index))  # cross_model is stored
        X = features.feature_matrix(frame)[features.FEATURES]
        for arm in arms:
            oof[arm][test] = eval05.fit_predict(arm, X[train], y[train], X[test], {})[0]  # no LLM arms here
        unseen[test] = ~ds.loc[test, "name"].isin(set(ds.loc[train, "name"])).to_numpy()
    return oof, unseen


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()
    config.load_dotenv()  # TABPFN_TOKEN for the first weight download
    ds = load_public()
    y = ds["y_ghost"].to_numpy(int)
    oof, unseen = run(ds)
    committed = pd.read_csv(config.RESULTS / "eval.csv").pivot_table(
        index=["split", "arm"], columns="metric", values="value")
    rows = []
    for split, mask in (("A", np.ones(len(ds), bool)), ("C", unseen)):
        for arm in ARMS:
            m = evaluate.metrics(y[mask], oof[arm][mask], probabilistic=True)
            for k in METRICS:
                rows.append({"split": split, "arm": arm, "metric": k, "public_rerun": m[k],
                             "committed": committed.loc[(split, arm), k]})
    table = pd.DataFrame(rows)
    table["diff"] = table["public_rerun"] - table["committed"]
    print(f"{len(ds)} rows, {int(y.sum())} ghosts; split C: {int(unseen.sum())} rows")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"max |diff| = {table['diff'].abs().max():.2e}")


if __name__ == "__main__":
    main()
