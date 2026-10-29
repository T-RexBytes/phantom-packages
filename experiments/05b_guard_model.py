"""Stage 05b: train the guard model (TabPFN-3.5-fast, name features only).

Same folds as results/folds.json. Writes results/model/guard_model.tabpfn_fit, guard_fold0..4,
guard_threshold.json, and a ``guard_model`` section in results/summary.json and run_config.json.
Deviation from BUILD_SPEC §9.1 (context features set to defaults): the guard never knows the
generation context, so it is trained without it; the numbers below justify the switch.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from rich.console import Console

from phantom import config, evaluate, model, runconfig

OOF_PATH = config.DATA / "oof_predictions.parquet"
console = Console(highlight=False)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    config.load_dotenv()
    import tabpfn
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import save_fitted_tabpfn_model

    ds = pd.read_parquet(config.DATASET_PATH)
    y = ds["y_ghost"].to_numpy(int)
    groups = ds["prompt_id"].to_numpy()
    folds = json.loads((config.RESULTS / "folds.json").read_text(encoding="utf-8"))["folds"]
    fold = ds["prompt_id"].map(folds).to_numpy()
    X = ds[model.GUARD_FEATURES].to_numpy(float)

    def make():
        return TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)

    oof = np.full(len(ds), np.nan)
    fit_s = predict_s = 0.0
    for k in range(evaluate.N_FOLDS):
        test = fold == k
        clf = make()
        t0 = time.perf_counter()
        clf.fit(X[~test], y[~test])
        t1 = time.perf_counter()
        oof[test] = clf.predict_proba(X[test])[:, 1]
        t2 = time.perf_counter()
        fit_s, predict_s = fit_s + t1 - t0, predict_s + t2 - t1
        save_fitted_tabpfn_model(clf, model.GUARD_FOLD_PATHS[k])
        console.print(f"  fold {k}: fit {t1 - t0:.1f}s predict {t2 - t1:.1f}s ({int(test.sum())} rows)")

    final = make()
    t0 = time.perf_counter()
    final.fit(X, y)
    final_fit_s = time.perf_counter() - t0
    save_fitted_tabpfn_model(final, model.GUARD_MODEL_PATH)

    m = evaluate.metrics(y, oof, probabilistic=True)
    m["auroc_lo"], m["auroc_hi"] = evaluate.bootstrap_auroc(y, oof, groups)
    threshold = evaluate.threshold_for_recall(y, oof)
    threshold.update(source="out-of-fold guard-model probabilities, split A folds", rule="flag ghost if p >= tau")
    runconfig.save(threshold, model.GUARD_THRESHOLD_PATH)

    phase5 = pd.read_parquet(OOF_PATH)
    aligned = (phase5[["name", "model", "prompt_id"]].to_numpy() == ds[["name", "model", "prompt_id"]].to_numpy()).all()
    if not aligned:
        raise SystemExit("data/oof_predictions.parquet is not row-aligned with the dataset; rerun stage eval")
    diff = evaluate.paired_auroc_diff(y, oof, phase5["A_tabpfn"].to_numpy(), groups)

    section = {
        "model": "TabPFN-3.5-fast (ModelVersion.V3_5_FAST), package defaults",
        "tabpfn_version": tabpfn.__version__,
        "features": model.GUARD_FEATURES,
        "oof_split_a": m,
        "threshold": threshold,
        "auroc_diff_vs_phase5_tabpfn": diff,
        "timing_s": {"fit_total": fit_s, "predict_total": predict_s, "final_fit": final_fit_s},
        "deviation": ("BUILD_SPEC §9.1 feeds the Phase-5 context model with default context "
                      "(consistency=1, cross_model=1, niche=2). The guard never knows the generation context, "
                      "so it uses TabPFN-3.5-fast on name features only. Measured on the same folds: "
                      "context model + defaults AUROC 0.971, ECE 0.029, 17 s per prediction (TabPFN-3.5, CPU); "
                      "name-only TabPFN-3.5 AUROC 0.973, ECE 0.014, 17 s; name-only 3.5-fast (this model) "
                      "about 3.5 s per prediction."),
    }
    summary = runconfig.load(config.SUMMARY_PATH) if config.SUMMARY_PATH.exists() else {}
    summary["guard_model"] = section
    runconfig.save(summary, config.SUMMARY_PATH)
    runconfig.update("guard_model", {"run_at": runconfig.utc_now(), **section})

    console.print(f"guard model OOF: AUROC {m['auroc']:.3f} [{m['auroc_lo']:.3f}, {m['auroc_hi']:.3f}] "
                  f"AUPRC {m['auprc']:.3f} Brier {m['brier']:.3f} ECE {m['ece']:.3f} "
                  f"P@R90 {m['precision_at_recall90']:.3f}")
    console.print(f"vs Phase-5 TabPFN (context features): {diff['diff']:+.3f} [{diff['lo']:+.3f}, {diff['hi']:+.3f}]")
    console.print(f"tau={threshold['tau']:.4f} recall={threshold['recall']:.3f} "
                  f"precision={threshold['precision']:.3f} flagged {threshold['n_flagged']}/{threshold['n']}")
    console.print(f"Wrote {model.GUARD_MODEL_PATH.relative_to(config.ROOT)}, 5 fold models, "
                  f"{model.GUARD_THRESHOLD_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
