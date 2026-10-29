"""Stage 08: public-safe site data (BUILD_SPEC §5.8) -> web/data.json and web/assets/.

Unregistered ghosts appear only as anonymised tombstones: keyed HMAC-SHA256 id (8 hex) plus
domain, niche level, mention count and models (CLAUDE.md rule 3). Registered ghosts (names
that were 404 at labelling but exist at the Phase 6 re-check) would be listed by name; the
squat-rule matches are published as a count only. tests/test_public_leak.py checks the output.
"""

from __future__ import annotations

import argparse
import json
import shutil

import pandas as pd

from phantom import config, privacy, prompts, runconfig

WEB = config.ROOT / "web"
ASSETS = WEB / "assets"
DATA_JSON = WEB / "data.json"
FIGURES = ["rates_by_model", "rates_by_niche", "calibration", "pr_curves", "repeat_hist"]
DEMOS = {"demo_hook_blocks_ghost.gif": "demo-hook.gif", "demo_offline_check.gif": "demo-offline.gif"}
ARMS = ["rule_top15k", "rule_editdist", "logreg", "hgb", "tabpfn", "llm_selfcheck"]
ARM_LABELS = {"rule_top15k": '"Not in top 15k" rule', "rule_editdist": "Edit-distance rule",
              "logreg": "Logistic regression", "hgb": "Gradient boosting", "tabpfn": "TabPFN-3.5",
              "llm_selfcheck": "gemma4 asks itself"}


def tombstones(labelled: pd.DataFrame, still_unregistered: set[str]) -> list[dict]:
    """One anonymised card per unregistered ghost, most-summoned first."""
    meta = {p["id"]: p for p in prompts.load_prompts()}
    g = labelled[labelled["y_ghost"] & labelled["name"].isin(still_unregistered)]
    rows = []
    for name, d in g.groupby("name"):
        pid = d["prompt_id"].iloc[0]  # every ghost came from a single prompt (checked in Phase 5)
        rows.append({"id": privacy.anon_id(name), "domain": meta[pid]["domain"],
                     "niche_level": int(meta[pid]["niche_level"]), "mentions": int(len(d)),
                     "models": sorted(d["model"].unique()), "n_prompts": int(d["prompt_id"].nunique())})
    return sorted(rows, key=lambda r: (-r["mentions"], -len(r["models"]), r["id"]))


def eval_table(summary: dict) -> dict:
    """split -> rows of the headline metrics per arm."""
    keys = ("auroc", "auroc_lo", "auroc_hi", "auprc", "precision_at_recall90", "recall_at_precision95", "brier", "ece")
    return {split: [{"arm": arm, "label": ARM_LABELS[arm],
                     **{k: summary["eval"][split][arm].get(k) for k in keys}} for arm in ARMS]
            for split in ("A", "B", "C")}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    labelled = pd.read_parquet(config.DATA / "labelled.parquet")
    squat = pd.read_csv(config.PRIVATE / "squat_full.csv")
    ghosts_at_label = squat[squat["why_candidate"].str.contains("ghost_at_label")]
    still = set(ghosts_at_label.loc[ghosts_at_label["exists_now"] == False, "name"])  # noqa: E712
    registered = ghosts_at_label[ghosts_at_label["exists_now"] == True]  # noqa: E712
    rates = summary["rates"]
    sq = summary["squat"]

    data = {
        "generated_at": runconfig.utc_now(),
        "source": "results/summary.json (every number on the site comes from there)",
        "headline": {
            "n_ghost_names": summary["repeat"]["n_unique_ghost_names"],
            "ghost_rate": rates["overall"]["ghost_rate"],
            "ghost_rate_lo": rates["overall"]["rate_lo"], "ghost_rate_hi": rates["overall"]["rate_hi"],
            "n_ghost_mentions": rates["overall"]["ghosts"], "n_mentions": rates["overall"]["mentions"],
            "gen_share": rates["overall"]["gen_share"],
            "repeat_share": summary["repeat"]["repeated_2plus_share"],
            "repeated_2plus": summary["repeat"]["repeated_2plus"],
            "multi_model": summary["repeat"]["multi_model"],
            "n_registered_during_study": sq["n_registered_during_study"],
            "n_squat_rule_matches": sq["n_possible_slopsquat"],
            "l3_over_l1": rates["l3_over_l1"],
        },
        "study": {k: summary["study"][k] for k in ("models", "n_prompts", "n_samples", "n_generations",
                                                    "n_generations_aborted", "generation_hours_total")},
        "per_model": {m: {k: r[k] for k in ("ghost_rate", "rate_lo", "rate_hi", "ghosts", "mentions", "gen_share")}
                      for m, r in rates["per_model"].items()},
        "per_niche": {lv: {k: r[k] for k in ("ghost_rate", "rate_lo", "rate_hi", "ghosts", "mentions")}
                      for lv, r in rates["per_niche_level"].items()},
        "eval": eval_table(summary),
        "loo": {k: v for k, v in summary["loo"].items() if k.endswith("|tabpfn")},
        "tabpfn_vs": {k: v for k, v in summary["auroc_diff_tabpfn_minus"].items()},
        "threshold": {k: summary["threshold"][k] for k in ("tau", "recall", "precision")},
        "guard_model": {"model": summary["guard_model"]["model"],
                        "auroc": summary["guard_model"]["oof_split_a"]["auroc"],
                        "ece": summary["guard_model"]["oof_split_a"]["ece"],
                        **{k: summary["guard_model"]["threshold"][k] for k in ("tau", "recall", "precision")}},
        "selfcheck": {k: summary["selfcheck"][k] for k in ("ghost_names", "ghost_names_said_exists",
                                                           "real_names", "real_names_said_missing")},
        "squat": {k: sq[k] for k in ("n_candidates", "n_ghosts_at_label", "n_still_unregistered",
                                     "n_registered_during_study", "n_possible_slopsquat",
                                     "n_born_after_summoning_names", "checked_at", "rule")},
        "registered_ghosts": [{"name": r["name"], "first_release": str(r["first_release"])[:10],
                               "n_releases": r["n_releases"]} for r in registered.to_dict("records")],
        "tombstones": tombstones(labelled, still),
    }
    WEB.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    DATA_JSON.write_text(json.dumps(data, indent=1, default=lambda v: v.item()) + "\n", encoding="utf-8")
    for f in FIGURES:
        shutil.copy2(config.RESULTS / f"{f}.png", ASSETS / f"{f}.png")
    demos = config.ROOT / "output"
    copied = [dst for src, dst in DEMOS.items() if (demos / src).exists()
              and shutil.copy2(demos / src, ASSETS / dst)]
    print(f"wrote {DATA_JSON.relative_to(config.ROOT)}: {len(data['tombstones'])} tombstones, "
          f"{len(data['registered_ghosts'])} registered ghosts; assets: {len(FIGURES)} figures, demos {copied}")


if __name__ == "__main__":
    main()
