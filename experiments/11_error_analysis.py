"""Error analysis: where does TabPFN go wrong at the guard threshold? (improvement item 3)

Inputs: data/oof_predictions.parquet (Phase-5 out-of-fold TabPFN, split A), data/dataset.parquet
(features), data/registry.parquet (registry facts: used here only to DESCRIBE mistakes, never as a
feature), results/model/threshold.json. Optional: data/guard_oof.parquet + guard_threshold.json.

False positives (real names flagged) get a registry-based category:
  young            first release within 24 months of the registry check
  registry_quiet   older, but < 500 downloads/month or <= 3 releases (barely used)
  established      older, >= 500 downloads/month and > 3 releases (real, just niche)
  no_files         exists on PyPI but has no released files (n_releases == 0)
False negatives (ghosts not flagged) get a name-shape category:
  py_affixed       py/python prefix or -py/py suffix around a word (looks like a binding)
  vendor_style     3+ hyphen parts (looks like an org-prefixed driver/plugin)
  short_word       a single short token (format, tool or protocol name)
  other
Online, an existing name is OK unless young (then SUSPECT), so "rescued_online" counts the false
positives the live guard would not block.

Outputs: results/error_analysis.json (aggregates only), summary.json["error_analysis"],
private/error_analysis.csv (per-row, with names: never public).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from phantom import config, features, runconfig

OOF_PATH = config.DATA / "oof_predictions.parquet"
GUARD_OOF_PATH = config.DATA / "guard_oof.parquet"
OUT_JSON = config.RESULTS / "error_analysis.json"
OUT_PRIVATE = config.PRIVATE / "error_analysis.csv"
YOUNG_DAYS = 30.44 * config.SQUAT_MAX_AGE_MONTHS
DESCRIBE = ["p", "nn_lev", "in_top15k", "affix_of_real", "char_lp_z", "consistency", "cross_model",
            "global_freq", "len", "n_parts"]


def downloads_bucket(value: Any) -> str:
    """Monthly-download bucket for a registry value (NaN/None -> 'unknown')."""
    if value is None or pd.isna(value):
        return "unknown"
    v = float(value)
    return "<100" if v < 100 else "100-499" if v < 500 else "500-9,999" if v < 10_000 else ">=10,000"


def fp_category(row: pd.Series, now: datetime) -> str:
    """Registry-based category of a real name that was flagged as a ghost."""
    if pd.isna(row.get("n_releases")) or int(row["n_releases"]) == 0 or pd.isna(row.get("first_release")):
        return "no_files"
    if now - pd.Timestamp(row["first_release"]).to_pydatetime() <= timedelta(days=YOUNG_DAYS):
        return "young"
    dl = row.get("recent_downloads_month")
    if int(row["n_releases"]) <= config.SQUAT_MAX_RELEASES or pd.isna(dl) or float(dl) < config.SQUAT_MAX_MONTHLY_DOWNLOADS:
        return "registry_quiet"
    return "established"


def fn_category(name: str, n_parts: int) -> str:
    """Name-shape category of a ghost the model let through."""
    bare = name.replace("-", "")
    if name.startswith(("py", "python-")) or (name.endswith(("-py", "py")) and len(bare) > 4):
        return "py_affixed"
    if n_parts >= 3:
        return "vendor_style"
    if n_parts == 1 and len(name) <= 10:
        return "short_word"
    return "other"


def with_oof_global_freq(ds: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """global_freq as each row saw it out of fold (training = the other folds)."""
    gf = pd.Series(np.nan, index=ds.index)
    for k in sorted(folds.unique()):
        test = folds == k
        gf[test] = features.with_global_freq(ds, ~test).loc[test, "global_freq"]
    return ds.assign(global_freq=gf)


def describe(rows: pd.DataFrame, extra: list[str] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(rows)), "names": int(rows["name"].nunique())}
    for c in DESCRIBE:
        if c in rows and len(rows):
            out[f"median_{c}"] = float(rows[c].median())
    for c in extra:
        out[c] = rows[c].value_counts().sort_index().to_dict()
    return out


def _year(years: pd.Series) -> float | None:
    """Median year, or None (valid JSON) when no row has a release date."""
    m = years.median()
    return None if pd.isna(m) else float(m)


def analyse(ds: pd.DataFrame, prob: pd.Series, tau: float, registry: pd.DataFrame, now: datetime
            ) -> tuple[dict[str, Any], pd.DataFrame]:
    """Aggregates and the per-row error table for one model's out-of-fold probabilities."""
    df = ds.assign(p=prob.to_numpy(), flagged=prob.to_numpy() >= tau)
    y = df["y_ghost"].astype(bool)
    conf = {k: {"rows": int(m.sum()), "names": int(df.loc[m, "name"].nunique())} for k, m in {
        "tp": df["flagged"] & y, "fp": df["flagged"] & ~y, "fn": ~df["flagged"] & y, "tn": ~df["flagged"] & ~y}.items()}

    fp = df[df["flagged"] & ~y].join(registry, on="name")
    fp["category"] = [fp_category(r, now) for _, r in fp.iterrows()]
    fp["downloads"] = fp["recent_downloads_month"].map(downloads_bucket)
    fp["first_release_year"] = pd.to_datetime(fp["first_release"], utc=True).dt.year
    fp["rescued_online"] = fp["category"] != "young"  # exists and not young -> the live guard says OK
    fn = df[~df["flagged"] & y].copy()
    fn["category"] = [fn_category(n, int(k)) for n, k in zip(fn["name"], fn["n_parts"])]

    fp_out = {"all": describe(fp, ["downloads"]) | {"in_top15k_share": float(fp["in_top15k"].mean()) if len(fp) else 0.0},
              "by_category": {c: describe(g, ["downloads"]) | {"median_first_release_year": _year(g["first_release_year"])}
                              for c, g in fp.groupby("category")},
              "rescued_online_rows": int(fp["rescued_online"].sum()),
              "rescued_online_names": int(fp.loc[fp["rescued_online"], "name"].nunique())}
    fn_out = {"all": describe(fn), "by_category": {c: describe(g) for c, g in fn.groupby("category")}}
    agg = {"tau": tau, "confusion": conf, "false_positives": fp_out, "false_negatives": fn_out}
    cols = ["name", "model", "prompt_id", "y_ghost", "p", "category", *[c for c in DESCRIBE if c != "p"]]
    table = pd.concat([fp.assign(error="FP")[cols + ["error", "first_release_year", "n_releases", "downloads",
                                                     "rescued_online"]],
                       fn.assign(error="FN")[cols + ["error"]]], ignore_index=True)
    return agg, table


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    oof = pd.read_parquet(OOF_PATH)
    ds = pd.read_parquet(config.DATASET_PATH)
    assert (oof[["name", "model", "prompt_id"]].to_numpy() == ds[["name", "model", "prompt_id"]].to_numpy()).all()
    ds = with_oof_global_freq(ds, oof["fold"])
    reg = pd.read_parquet(config.REGISTRY_PATH).set_index("name")
    now = pd.to_datetime(reg["checked_at"], utc=True).max().to_pydatetime()
    registry = reg[["first_release", "n_releases", "recent_downloads_month"]]

    tau = json.loads((config.MODEL_DIR / "threshold.json").read_text(encoding="utf-8"))["tau"]
    agg, table = analyse(ds, oof["A_tabpfn"], tau, registry, now)
    result: dict[str, Any] = {"generated_at": runconfig.utc_now(), "registry_checked_at": now.isoformat(),
                              "phase5_tabpfn_split_a": agg}
    tables = [table.assign(source="phase5_tabpfn")]

    if GUARD_OOF_PATH.exists():
        g = pd.read_parquet(GUARD_OOF_PATH)
        pcol = next(c for c in ("p_ghost", "guard_p", "p", "A_guard") if c in g.columns)
        g = ds.merge(g[["model", "prompt_id", "name", pcol]], on=["model", "prompt_id", "name"], how="left")
        gtau = json.loads((config.MODEL_DIR / "guard_threshold.json").read_text(encoding="utf-8"))["tau"]
        gagg, gtable = analyse(ds, g[pcol], gtau, registry, now)
        result["guard_model_split_a"] = gagg
        tables.append(gtable.assign(source="guard_model"))
    else:
        result["guard_model_split_a"] = "skipped: data/guard_oof.parquet not found"

    runconfig.save(result, OUT_JSON)
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))  # reload right before saving
    summary["error_analysis"] = result
    runconfig.save(summary, config.SUMMARY_PATH)
    config.PRIVATE.mkdir(exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(OUT_PRIVATE, index=False)

    c, fpo = agg["confusion"], agg["false_positives"]
    print(f"tau={tau:.4f}  TP {c['tp']}  FP {c['fp']}  FN {c['fn']}  TN {c['tn']}")
    print("FP categories:", {k: v["rows"] for k, v in fpo["by_category"].items()},
          f"| rescued online: {fpo['rescued_online_rows']} rows")
    print("FN categories:", {k: v["rows"] for k, v in agg["false_negatives"]["by_category"].items()})
    print(f"guard model: {'done' if GUARD_OOF_PATH.exists() else result['guard_model_split_a']}")
    print(f"wrote {OUT_JSON.relative_to(config.ROOT)}, summary.json[error_analysis], "
          f"{OUT_PRIVATE.relative_to(config.ROOT)} (private)")


if __name__ == "__main__":
    main()
