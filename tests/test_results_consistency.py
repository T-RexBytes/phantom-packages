"""Committed results agree with each other (BUILD_SPEC §8, §12): every headline number in
summary.json matches the file it came from, counts add up, and every CI brackets its point.
Runs in CI on committed files only (no private data, no models)."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from phantom import config, privacy

R = config.RESULTS


def load(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def s() -> dict:
    return load("summary.json")


def test_rates_agree_across_files(s):
    ls = load("label_summary.json")
    rates = pd.read_csv(R / "rates.csv").set_index(["scope", "group"])
    overall = s["rates"]["overall"]
    assert overall["ghosts"] == ls["n_ghost_mentions"] == rates.loc[("overall", "all"), "ghosts"]
    assert overall["ghost_rate"] == pytest.approx(ls["ghost_rate_overall"])
    assert overall["ghost_rate"] == pytest.approx(overall["ghosts"] / overall["mentions"])
    for scope, key in (("model", "per_model"), ("niche_level", "per_niche_level")):
        part = rates.loc[scope]
        assert part["ghosts"].sum() == overall["ghosts"] and part["mentions"].sum() == overall["mentions"]
        for group, r in s["rates"][key].items():
            assert r["ghosts"] == part.loc[group, "ghosts"]
    for r in [overall, *s["rates"]["per_model"].values(), *s["rates"]["per_niche_level"].values()]:
        assert r["rate_lo"] <= r["ghost_rate"] <= r["rate_hi"]
        assert r["gen_share_lo"] <= r["gen_share"] <= r["gen_share_hi"]


def test_study_counts(s):
    st = s["study"]
    assert st["n_generations"] + st["n_generations_aborted"] == st["n_generations_planned"]
    assert st["n_generations_planned"] == st["n_models"] * st["n_prompts"] * st["n_samples"]


def test_eval_table_is_complete_and_brackets(s):
    ev = pd.read_csv(R / "eval.csv")
    wide = ev.pivot_table(index=["split", "arm"], columns="metric", values="value")
    required = {"rule_top15k", "rule_editdist", "logreg", "hgb", "tabpfn", "llm_selfcheck", "llm_selfcheck_prob"}
    optional = {"hosted_selfcheck", "hosted_selfcheck_prob"}  # present once the hosted self-check is cached
    arms = set(wide.index.get_level_values("arm"))
    assert required <= arms <= required | optional
    assert set(wide.index) == {(sp, a) for sp in "ABC" for a in arms}
    assert (wide["auroc_lo"] <= wide["auroc"]).all() and (wide["auroc"] <= wide["auroc_hi"]).all()
    for (sp, arm), row in wide.iterrows():
        assert s["eval"][sp][arm]["auroc"] == pytest.approx(row["auroc"])
    loo = pd.read_csv(R / "loo.csv")
    assert len(loo) == s["study"]["n_models"] * wide.index.get_level_values("arm").nunique()
    assert loo.groupby("held_out_model")["n"].first().sum() == s["dataset"]["n_rows"]


def test_calibration_bins_cover_every_row(s):
    cal = pd.read_csv(R / "calibration.csv")
    assert (cal.groupby("arm")["n"].sum() == s["dataset"]["n_rows"]).all()


def test_thresholds_meet_the_recall_target(s):
    for name in ("model/threshold.json", "model/guard_threshold.json"):
        t = load(name)
        assert t["recall"] >= t["target_recall"] and 0 < t["tau"] < 1
    assert s["threshold"]["tau"] == load("model/threshold.json")["tau"]
    assert s["guard_model"]["threshold"]["tau"] == load("model/guard_threshold.json")["tau"]


def test_repeat_and_squat_counts(s):
    rep = pd.read_csv(R / "repeat_counts.csv")
    assert rep["ghost_names"].sum() == s["repeat"]["n_unique_ghost_names"]
    assert (rep["mentions"] * rep["ghost_names"]).sum() == s["rates"]["overall"]["ghosts"]
    assert rep.loc[rep["mentions"] >= 2, "ghost_names"].sum() == s["repeat"]["repeated_2plus"]
    assert rep["by_2plus_models"].sum() == s["repeat"]["multi_model"]
    sq = load("squat_summary.json")
    assert sq == s["squat"]
    assert sq["n_still_unregistered"] + sq["n_registered_during_study"] == sq["n_ghosts_at_label"]
    assert len(privacy.load_anon_ids()) == sq["n_still_unregistered"]
    pub = pd.read_csv(R / "squat_public.csv")
    assert "name" not in pub.columns and len(pub) == sq["n_registered_candidates"]
    assert int(pub["possible_slopsquat"].sum()) == sq["n_possible_slopsquat"]


def test_site_data_matches_summary(s):
    d = json.loads((config.ROOT / "web" / "data.json").read_text(encoding="utf-8"))
    assert d["headline"]["n_ghost_mentions"] == s["rates"]["overall"]["ghosts"]
    assert len(d["tombstones"]) == s["squat"]["n_still_unregistered"]
    assert sum(t["mentions"] for t in d["tombstones"]) == s["rates"]["overall"]["ghosts"]
    assert {t["id"] for t in d["tombstones"]} == privacy.load_anon_ids()


def test_readme_numbers_come_from_summary(s):
    readme = (config.ROOT / "README.md").read_text(encoding="utf-8")
    o, g = s["rates"]["overall"], s["guard_model"]["threshold"]
    assert f"{o['ghosts']} of the {o['mentions']:,}" in readme and f"({o['ghost_rate']:.1%})" in readme
    assert f"τ = {g['tau']:.3f}" in readme
    assert f"{g['recall']:.1%} of ghosts" in readme and f"{g['precision']:.1%} of the names" in readme
    for pct in re.findall(r"\d+\.\d%", readme):  # no stray percentages that aren't in the summary
        assert pct in {f"{o['ghost_rate']:.1%}", f"{g['recall']:.1%}", f"{g['precision']:.1%}"}, pct
