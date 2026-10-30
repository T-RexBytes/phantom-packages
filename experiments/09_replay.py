"""Stage 09: replay every generated install through the guard ("what if my agent had run them all?").

For each of the in-plan generations, the install names the model wrote (``pip install`` lines)
go through the same verdict logic as ``phantom.guard`` (and so the Claude Code hook):
  online   stdlib -> OK; 404 on PyPI -> GHOST; exists -> SUSPECT if the squat rule (minus
           "repeated", which needs study data) or (P(ghost) >= tau and young); else OK
  offline  stdlib -> OK; P(ghost) >= tau -> GHOST_LIKELY; else OK_UNVERIFIED
PyPI's answer is the label-time registry (data/registry.parquet), so this is cache-only.
P(ghost) is always out-of-fold: each name is scored by the guard fold model whose training
fold excluded that prompt (results/folds.json), never by a model that saw the row.
GHOST, SUSPECT and GHOST_LIKELY are what the hook blocks.

Outputs: results/replay.json (aggregates) and summary.json["replay"]; per-name false blocks
go to private/replay_false_blocks.csv (never public).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from phantom import config, generate, model, runconfig, squat
from phantom.extract import STDLIB, normalize

INSTALL_SOURCES = ("pip_line", "uv_line", "requirements_block")  # what an agent would actually run
GUARD_SQUAT_CHECKS = [c for c in squat.CHECKS if c != "repeated"]  # as in phantom.guard
BLOCKING = {"GHOST", "SUSPECT", "GHOST_LIKELY"}
REPLAY_PATH = config.RESULTS / "replay.json"
FALSE_BLOCKS_PATH = config.PRIVATE / "replay_false_blocks.csv"
GUARD_OOF_PATH = config.DATA / "guard_oof.parquet"  # out-of-fold guard P(ghost) per dataset row (private: names)
_STDLIB = frozenset(normalize(m) for m in STDLIB)
console = Console(highlight=False)


def online_verdict(name: str, row: dict | None, p: float, tau: float) -> tuple[str, str]:
    """(verdict, reason) as phantom.guard decides it with PyPI reachable."""
    if name in _STDLIB:
        return "OK", "standard library"
    if row is None or not row["exists"]:
        return "GHOST", "404 on PyPI"
    now = pd.Timestamp(row["checked_at"]).to_pydatetime() if row.get("checked_at") is not None else None
    c = squat.checks(row, 0, now)
    if all(c[k] for k in GUARD_SQUAT_CHECKS):
        return "SUSPECT", "squat rule (few releases, young, low downloads)"
    if not np.isnan(p) and p >= tau and c["young"]:
        return "SUSPECT", "young and ghost-like name"
    return "OK", "exists on PyPI"


def offline_verdict(name: str, p: float, tau: float) -> str:
    """Verdict with PyPI unreachable or --offline: the model's estimate alone."""
    if name in _STDLIB:
        return "OK"
    return "GHOST_LIKELY" if not np.isnan(p) and p >= tau else "OK_UNVERIFIED"


def oof_probabilities(pairs: pd.DataFrame, folds: dict[str, int]) -> pd.Series:
    """P(ghost) per (prompt_id, name) from the guard fold model that never saw that prompt."""
    pairs = pairs.assign(fold=pairs["prompt_id"].map(folds))
    out = pd.Series(np.nan, index=pd.MultiIndex.from_frame(pairs[["prompt_id", "name"]]))
    for k, part in pairs.groupby("fold"):
        names = sorted(part["name"].unique())
        est = model.load_estimator(model.GUARD_FOLD_PATHS[int(k)])
        p = dict(zip(names, est.predict_proba(model.name_feature_frame(names).to_numpy(float))[:, 1]))
        for pid, n in zip(part["prompt_id"], part["name"]):
            out[(pid, n)] = p[n]
    return out


def summarize(m: pd.DataFrame, gens: pd.DataFrame) -> dict:
    """Aggregates for one slice: m has one row per (generation, name) with both verdicts."""
    real, ghost = m["category"].eq("real"), m["category"].eq("ghost")
    sub = m["category"].isin(["stdlib_confusion", "module_name_confusion"])
    on, off = m["online"].isin(BLOCKING), m["offline"].isin(BLOCKING)
    blocked_gens_on = m.loc[on, "key"].nunique()
    blocked_gens_off = m.loc[off, "key"].nunique()
    with_install = m["key"].nunique()
    return {
        "generations": int(len(gens)),
        "generations_with_install": int(with_install),
        "install_mentions": int(len(m)),
        "ghost_mentions": int(ghost.sum()),
        "real_mentions": int(real.sum()),
        "subtype_mentions": {c: {v: int(n) for v, n in m.loc[m["category"].eq(c), "online"].value_counts().items()}
                             for c in ("stdlib_confusion", "module_name_confusion")},
        "subtype_total": int(sub.sum()),
        "online": {
            "generations_blocked": int(blocked_gens_on),
            "generations_blocked_share": blocked_gens_on / max(with_install, 1),
            "ghosts_blocked": int((ghost & on).sum()),
            "real_blocked": int((real & on).sum()),
            "real_blocked_share": float((real & on).sum() / max(real.sum(), 1)),
            "suspect_mentions": int(m["online"].eq("SUSPECT").sum()),
        },
        "offline": {
            "generations_blocked": int(blocked_gens_off),
            "ghosts_caught": int((ghost & off).sum()),
            "ghosts_missed": int((ghost & ~off).sum()),
            "ghost_recall": float((ghost & off).sum() / max(ghost.sum(), 1)),
            "real_flagged": int((real & off).sum()),
            "real_flagged_share": float((real & off).sum() / max(real.sum(), 1)),
            "precision": float((ghost & off).sum() / max(off[real | ghost].sum(), 1)),
        },
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    summon = runconfig.load()["summon"]
    records = generate.in_plan(generate.read_records(config.RAW_GENERATIONS_PATH), summon)
    gens = pd.DataFrame([{"key": r["key"], "model": r["model"]} for r in records if r.get("arm") != "hosted"])
    lab = pd.read_parquet(config.DATA / "labelled.parquet")
    lab = lab[lab["source"].isin(INSTALL_SOURCES) & lab["key"].isin(set(gens["key"]))]
    m = lab.drop_duplicates(["key", "name"])[["key", "model", "prompt_id", "name", "category"]].copy()

    reg = pd.read_parquet(config.REGISTRY_PATH)
    rows = {r["name"]: r for r in reg.to_dict("records")}
    folds = json.loads((config.RESULTS / "folds.json").read_text(encoding="utf-8"))["folds"]
    tau = model.guard_tau()
    p = oof_probabilities(m[["prompt_id", "name"]].drop_duplicates(), folds)
    m["p_guard"] = [p[(pid, n)] for pid, n in zip(m["prompt_id"], m["name"])]
    ds = pd.read_parquet(config.DATASET_PATH)  # also save guard OOF per dataset row (stage 11 reads it)
    pds = oof_probabilities(ds[["prompt_id", "name"]].drop_duplicates(), folds)
    ds.assign(p_ghost=[pds[(pid, n)] for pid, n in zip(ds["prompt_id"], ds["name"])])[
        ["model", "prompt_id", "name", "y_ghost", "p_ghost"]].to_parquet(GUARD_OOF_PATH, index=False)
    verdicts = [online_verdict(n, rows.get(n), pg, tau) for n, pg in zip(m["name"], m["p_guard"])]
    m["online"], m["reason"] = [v for v, _ in verdicts], [r for _, r in verdicts]
    m["offline"] = [offline_verdict(n, pg, tau) for n, pg in zip(m["name"], m["p_guard"])]

    result = {"run_at": runconfig.utc_now(), "tau": tau, "registry": "label-time PyPI answers (data/registry.parquet)",
              "p_ghost": "out-of-fold guard fold models (folds.json)", "install_sources": list(INSTALL_SOURCES),
              "overall": summarize(m, gens),
              "per_model": {mod: summarize(m[m["model"] == mod], gens[gens["model"] == mod])
                            for mod in summon["models"]}}
    runconfig.save(result, REPLAY_PATH)
    summary = runconfig.load(config.SUMMARY_PATH) if config.SUMMARY_PATH.exists() else {}  # reload: other writers
    summary["replay"] = result
    runconfig.save(summary, config.SUMMARY_PATH)

    false_blocks = m[m["category"].eq("real") & (m["online"].isin(BLOCKING) | m["offline"].isin(BLOCKING))]
    config.PRIVATE.mkdir(parents=True, exist_ok=True)
    (false_blocks.groupby(["name", "online", "offline", "reason"])
     .agg(mentions=("key", "size"), models=("model", lambda s: ",".join(sorted(set(s)))),
          p_guard_max=("p_guard", "max"))
     .reset_index().sort_values("mentions", ascending=False).to_csv(FALSE_BLOCKS_PATH, index=False))

    table = Table(title=f"Replay: every generated install through the guard (tau={tau:.3f})")
    for col in ("slice", "gens w/ install", "online blocked", "ghosts blocked", "real wrongly blocked",
                "offline: ghosts caught", "offline: real flagged"):
        table.add_column(col, justify="left" if col == "slice" else "right")
    for name, r in [("all", result["overall"]), *result["per_model"].items()]:
        on, off = r["online"], r["offline"]
        table.add_row(name, str(r["generations_with_install"]),
                      f"{on['generations_blocked']} ({on['generations_blocked_share']:.1%})",
                      f"{on['ghosts_blocked']}/{r['ghost_mentions']}",
                      f"{on['real_blocked']}/{r['real_mentions']} ({on['real_blocked_share']:.1%})",
                      f"{off['ghosts_caught']}/{r['ghost_mentions']} ({off['ghost_recall']:.1%})",
                      f"{off['real_flagged']}/{r['real_mentions']} ({off['real_flagged_share']:.1%})")
    console.print(table)
    console.print(f"subtypes (online verdicts): {result['overall']['subtype_mentions']}")
    console.print(f"Wrote {REPLAY_PATH.relative_to(config.ROOT)} and summary.json['replay']; "
                  f"per-name false blocks in {FALSE_BLOCKS_PATH.relative_to(config.ROOT)} (private)")


if __name__ == "__main__":
    main()
