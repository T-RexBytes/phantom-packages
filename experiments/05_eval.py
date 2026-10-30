"""Stage 05: evaluate six arms on three splits (BUILD_SPEC §8).

Splits (same folds for every arm, saved to results/folds.json):
  A  GroupKFold(5) by prompt_id
  B  leave-one-model-out (cross_model/global_freq from the training models only;
     the `model` feature is dropped because the held-out model's code is unseen)
  C  rows of A whose name never appears in their training fold

Outputs: results/eval.csv, loo.csv, rates.csv, calibration.csv, folds.json,
results/model/{tabpfn_model,tabpfn_fold0..4}.tabpfn_fit, threshold.json, summary.json.
Per-row predictions contain names, so they go to data/oof_predictions.parquet (private).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from phantom import config, evaluate, features, generate, hosted, prompts, runconfig, selfcheck

ARMS = ["rule_top15k", "rule_editdist", "logreg", "hgb", "tabpfn", "llm_selfcheck", "llm_selfcheck_prob"]
PROBABILISTIC = {"rule_top15k": True, "rule_editdist": False, "logreg": True, "hgb": True,
                 "tabpfn": True, "llm_selfcheck": True, "llm_selfcheck_prob": True}
# llm_selfcheck: the spec's hard yes/no; llm_selfcheck_prob: P(no)/(P(no)+P(yes)) from gemma's top logprobs
# hosted_selfcheck(_prob): the same question to Ultramarine (hosted, not open); added only when every
# dataset name has a cached answer (missing ones are asked under the hosted spend ledger if a key is set).
HOSTED_ARMS = ["hosted_selfcheck", "hosted_selfcheck_prob"]
RATES_HOSTED_PATH = config.RESULTS / "rates_hosted.csv"
LABELLED_PATH = config.DATA / "labelled.parquet"
GENERATIONS_PATH = config.DATA / "generations.parquet"
OOF_PATH = config.DATA / "oof_predictions.parquet"
FOLDS_PATH = config.RESULTS / "folds.json"
EVAL_PATH = config.RESULTS / "eval.csv"
LOO_PATH = config.RESULTS / "loo.csv"
RATES_PATH = config.RESULTS / "rates.csv"
CALIBRATION_PATH = config.RESULTS / "calibration.csv"
DIFF_PATH = config.RESULTS / "auroc_diff.csv"
THRESHOLD_PATH = config.MODEL_DIR / "threshold.json"
FINAL_MODEL_PATH = config.MODEL_DIR / "tabpfn_model.tabpfn_fit"
console = Console(highlight=False)


# ---------------------------------------------------------------- arms

def _categorical(cols: list[str]) -> list[str]:
    return [c for c in features.CATEGORICAL if c in cols]


def make_tabpfn(cols: list[str]):
    """TabPFNClassifier with package defaults (TabPFN-3.5), told which columns are categorical."""
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(categorical_features_indices=[cols.index(c) for c in _categorical(cols)])


def make_model(arm: str, cols: list[str]):
    """Untrained estimator for a learned arm."""
    cats = _categorical(cols)
    if arm == "logreg":
        nums = [c for c in cols if c not in cats]
        prep = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ])
        return make_pipeline(prep, LogisticRegression(class_weight="balanced", max_iter=2000))
    if arm == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=config.GLOBAL_SEED,
                                              categorical_features=[cols.index(c) for c in cats])
    if arm == "tabpfn":
        return make_tabpfn(cols)
    raise ValueError(arm)


def fit_predict(arm: str, Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame,
                llm_scores: dict[str, np.ndarray]) -> tuple[np.ndarray, float, float, Any]:
    """Scores for the test rows, fit seconds, predict seconds, fitted object (or tuned tau)."""
    if arm == "rule_top15k":
        return 1.0 - Xte["in_top15k"].to_numpy(float), 0.0, 0.0, None
    if arm == "rule_editdist":
        t0 = time.perf_counter()
        tau = evaluate.threshold_for_recall(ytr, Xtr["nn_lev"].to_numpy(float))["tau"]  # tuned on train
        return Xte["nn_lev"].to_numpy(float), time.perf_counter() - t0, 0.0, tau
    if arm in llm_scores:  # no training: the model's own answer for the test rows
        return llm_scores[arm], 0.0, 0.0, None
    model = make_model(arm, list(Xtr.columns))
    t0 = time.perf_counter()
    model.fit(Xtr.to_numpy(float) if arm == "tabpfn" else Xtr, ytr)
    t1 = time.perf_counter()
    prob = model.predict_proba(Xte.to_numpy(float) if arm == "tabpfn" else Xte)[:, 1]
    return prob, t1 - t0, time.perf_counter() - t1, model


# ---------------------------------------------------------------- splits

def run_fold(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray, cols: list[str],
             sc: dict[str, np.ndarray], label: str) -> tuple[dict[str, np.ndarray], list[dict], dict[str, Any]]:
    """All arms on one train/test split."""
    X = features.feature_matrix(frame)[cols]
    y = frame["y_ghost"].to_numpy(int)
    scores, timing, fitted = {}, [], {}
    for arm in ARMS:
        s, fit_s, pred_s, obj = fit_predict(arm, X[train], y[train], X[test], {a: v[test] for a, v in sc.items()})
        scores[arm], fitted[arm] = s, obj
        timing.append({"fold": label, "arm": arm, "fit_s": fit_s, "predict_s": pred_s,
                       "n_train": int(train.sum()), "n_test": int(test.sum())})
        console.print(f"  {label:<22} {arm:<14} fit {fit_s:6.1f}s predict {pred_s:6.1f}s")
    return scores, timing, fitted


def metric_rows(split: str, y: np.ndarray, groups: np.ndarray, scores: dict[str, np.ndarray],
                extra: dict[str, dict[str, float]] | None = None) -> list[dict]:
    """Long-format metric rows with the bootstrap AUROC CI."""
    rows = []
    for arm in ARMS:
        m = evaluate.metrics(y, scores[arm], PROBABILISTIC[arm])
        m["auroc_lo"], m["auroc_hi"] = evaluate.bootstrap_auroc(y, scores[arm], groups)
        m.update((extra or {}).get(arm, {}))
        rows += [{"split": split, "arm": arm, "metric": k, "value": v} for k, v in m.items()]
    return rows


# ---------------------------------------------------------------- rates

def rate_rows(labelled: pd.DataFrame, gens: pd.DataFrame, niche: dict[str, int]) -> pd.DataFrame:
    """Ghost rates (mentions) and generation-level shares with prompt-bootstrap CIs."""
    main = labelled[labelled["category"].isin(["real", "ghost"])].assign(niche=lambda d: d["prompt_id"].map(niche))
    ghost_keys = set(main.loc[main["y_ghost"], "key"])
    g = gens.assign(has_ghost=gens["key"].isin(ghost_keys), niche=gens["prompt_id"].map(niche))
    rows = []
    for scope, col in (("overall", None), ("model", "model"), ("niche_level", "niche")):
        groups = [("all", main, g)] if col is None else [
            (f"L{v}" if col == "niche" else v, main[main[col] == v], g[g[col] == v]) for v in sorted(main[col].unique())]
        for name, m, gg in groups:
            rate, lo, hi = evaluate.bootstrap_rate(m["y_ghost"].to_numpy(), m["prompt_id"].to_numpy())
            share, slo, shi = evaluate.bootstrap_rate(gg["has_ghost"].to_numpy(), gg["prompt_id"].to_numpy())
            rows.append({"scope": scope, "group": name, "mentions": len(m), "ghosts": int(m["y_ghost"].sum()),
                         "ghost_rate": rate, "rate_lo": lo, "rate_hi": hi, "generations": len(gg),
                         "with_ghost": int(gg["has_ghost"].sum()), "gen_share": share,
                         "gen_share_lo": slo, "gen_share_hi": shi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    config.load_dotenv()  # TABPFN_TOKEN / TABPFN_NO_BROWSER for the first weights download
    import tabpfn
    import torch
    from tabpfn.model_loading import save_fitted_tabpfn_model
    from tabpfn.settings import settings as tabpfn_settings

    ds = pd.read_parquet(config.DATASET_PATH)
    y = ds["y_ghost"].to_numpy(int)
    groups = ds["prompt_id"].to_numpy()
    models = sorted(ds["model"].unique())

    folds = evaluate.assign_folds(ds["prompt_id"])
    runconfig.save({"method": "GroupKFold(n_splits=5, shuffle=True, random_state=42) by prompt_id",
                    "n_folds": evaluate.N_FOLDS, "folds": folds}, FOLDS_PATH)
    ds["fold"] = ds["prompt_id"].map(folds)

    console.rule(f"llm_selfcheck: asking {selfcheck.MODEL} about {ds['name'].nunique()} names (cached)")
    show = lambda i, n: i % 100 == 0 and console.print(f"  {i}/{n}")  # noqa: E731
    answers = selfcheck.ask_all(ds["name"], progress=show)
    answers_prob = selfcheck.ask_all(ds["name"], progress=show, top_k=selfcheck.TOP_K)
    ds["selfcheck_answer"] = ds["name"].map(lambda n: answers[n]["answer"])
    sc = {"llm_selfcheck": ds["selfcheck_answer"].map(selfcheck.score).to_numpy(float),
          "llm_selfcheck_prob": ds["name"].map(lambda n: selfcheck.prob_score(answers_prob[n])).to_numpy(float)}
    selfcheck_s = sum(answers[n]["latency_s"] for n in ds["name"].unique())
    hosted_answers = hosted_selfcheck_answers(ds["name"])
    if hosted_answers is not None:
        sc["hosted_selfcheck"] = ds["name"].map(lambda n: selfcheck.score(hosted_answers[n]["answer"])).to_numpy(float)
        sc["hosted_selfcheck_prob"] = ds["name"].map(lambda n: selfcheck.prob_score(hosted_answers[n])).to_numpy(float)
        ARMS.extend(a for a in HOSTED_ARMS if a not in ARMS)
        PROBABILISTIC.update({a: True for a in HOSTED_ARMS})

    all_cols = list(features.FEATURES)
    oof = {arm: np.full(len(ds), np.nan) for arm in ARMS}
    timing: list[dict] = []
    tuned = []  # (fold, tau, test mask) for rule_editdist
    unseen = np.zeros(len(ds), bool)

    console.rule("Split A: GroupKFold(5) by prompt")
    for k in range(evaluate.N_FOLDS):
        test = (ds["fold"] == k).to_numpy()
        train = ~test
        frame = features.with_global_freq(evaluate.with_cross_model(ds, set(models)), pd.Series(train, index=ds.index))
        scores, t, fitted = run_fold(frame, train, test, all_cols, sc, f"A fold {k}")
        timing += [{**r, "split": "A"} for r in t]
        for arm in ARMS:
            oof[arm][test] = scores[arm]
        tuned.append((fitted["rule_editdist"], test))
        unseen[test] = ~ds.loc[test, "name"].isin(set(ds.loc[train, "name"])).to_numpy()
        save_fitted_tabpfn_model(fitted["tabpfn"], config.MODEL_DIR / f"tabpfn_fold{k}.tabpfn_fit")

    def tuned_extra(mask: np.ndarray) -> dict[str, dict[str, float]]:
        flagged = np.zeros(len(ds), bool)
        for tau, test in tuned:
            flagged[test] = ds.loc[test, "nn_lev"].to_numpy() >= tau
        f, yy = flagged[mask], y[mask]
        tp = int((f & (yy == 1)).sum())
        return {"rule_editdist": {"tuned_tau_mean": float(np.mean([t for t, _ in tuned])),
                                  "recall_at_tuned_tau": tp / max(int(yy.sum()), 1),
                                  "precision_at_tuned_tau": tp / max(int(f.sum()), 1)}}

    eval_rows = metric_rows("A", y, groups, oof, tuned_extra(np.ones(len(ds), bool)))
    eval_rows += metric_rows("C", y[unseen], groups[unseen], {a: s[unseen] for a, s in oof.items()},
                             tuned_extra(unseen))

    console.rule("Split B: leave-one-model-out")
    b_cols = [c for c in all_cols if c != "model"]
    loo = {arm: np.full(len(ds), np.nan) for arm in ARMS}
    loo_rows = []
    for held in models:
        test = (ds["model"] == held).to_numpy()
        train = ~test
        known = set(models) - {held}
        frame = features.with_global_freq(evaluate.with_cross_model(ds, known), pd.Series(train, index=ds.index))
        scores, t, _ = run_fold(frame, train, test, b_cols, sc, f"B held-out {held}")
        timing += [{**r, "split": "B"} for r in t]
        for arm in ARMS:
            loo[arm][test] = scores[arm]
            m = evaluate.metrics(y[test], scores[arm], PROBABILISTIC[arm])
            m["auroc_lo"], m["auroc_hi"] = evaluate.bootstrap_auroc(y[test], scores[arm], groups[test])
            loo_rows.append({"held_out_model": held, "arm": arm, **m})
    eval_rows += metric_rows("B", y, groups, loo)

    diff_rows = [{"split": split, "arm": arm, **evaluate.paired_auroc_diff(y[m], s["tabpfn"][m], s[arm][m], groups[m])}
                 for split, s, m in (("A", oof, np.ones(len(ds), bool)), ("B", loo, np.ones(len(ds), bool)),
                                     ("C", oof, unseen))
                 for arm in ARMS if arm != "tabpfn"]
    pd.DataFrame(diff_rows).to_csv(DIFF_PATH, index=False)
    console.print("AUROC(tabpfn) - AUROC(arm), paired prompt bootstrap:\n"
                  + pd.DataFrame(diff_rows).round(3).to_string(index=False))

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(EVAL_PATH, index=False)
    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(LOO_PATH, index=False)
    timing_df = pd.DataFrame(timing)

    calib = pd.concat([evaluate.calibration_table(y, oof[arm]).assign(arm=arm)
                       for arm in ARMS if PROBABILISTIC[arm]])
    calib.to_csv(CALIBRATION_PATH, index=False)

    threshold = evaluate.threshold_for_recall(y, oof["tabpfn"])
    threshold.update(source="out-of-fold TabPFN probabilities, split A", rule="flag ghost if p >= tau")
    runconfig.save(threshold, THRESHOLD_PATH)

    console.rule("Final TabPFN model on all rows")
    final = features.with_global_freq(evaluate.with_cross_model(ds, set(models)), pd.Series(True, index=ds.index))
    clf = make_tabpfn(all_cols)
    t0 = time.perf_counter()
    clf.fit(features.feature_matrix(final)[all_cols].to_numpy(float), y)
    final_fit_s = time.perf_counter() - t0
    save_fitted_tabpfn_model(clf, FINAL_MODEL_PATH)

    pd.DataFrame({"name": ds["name"], "model": ds["model"], "prompt_id": ds["prompt_id"], "fold": ds["fold"],
                  "y_ghost": y, "unseen_name": unseen, "selfcheck_answer": ds["selfcheck_answer"],
                  **{f"A_{a}": s for a, s in oof.items()}, **{f"B_{a}": s for a, s in loo.items()}}
                 ).to_parquet(OOF_PATH, index=False)

    # --- rates (with prompt-bootstrap CIs) and the summary
    labelled = pd.read_parquet(LABELLED_PATH)
    gens = pd.read_parquet(GENERATIONS_PATH)
    niche = {p["id"]: p["niche_level"] for p in prompts.load_prompts()}
    rates = rate_rows(labelled, gens, niche)
    rates.to_csv(RATES_PATH, index=False)
    write_summary(ds, y, unseen, eval_df, loo_df, timing_df, rates, threshold, answers, answers_prob, selfcheck_s,
                  final_fit_s, diff_rows, {
                      "tabpfn_version": tabpfn.__version__, "model_version": str(tabpfn_settings.tabpfn.model_version.value),
                      "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__})
    add_hosted_summary(niche, ds, hosted_answers)
    print_tables(eval_df, loo_df, timing_df, threshold)


def hosted_selfcheck_answers(names) -> dict | None:
    """Ultramarine self-check answers for every name, or None (arms skipped) if any is missing and
    cannot be asked (no key, or the hosted spend cap would be exceeded)."""
    names = sorted(set(names))
    cached = hosted.cached_selfcheck(names)
    missing = [n for n in names if n not in cached]
    if not missing:
        return cached
    client = hosted.client_from_env()
    if client is None:
        if cached:
            console.print(f"[yellow]hosted_selfcheck skipped: {len(missing)} names unanswered and no key")
        return None
    console.rule(f"hosted_selfcheck: asking {hosted.MODEL} about {len(missing)} new names")
    try:
        return hosted.ask_selfcheck(names, client, hosted.fetch_pricing(), hosted.Ledger())
    except hosted.BudgetExceeded as exc:
        console.print(f"[yellow]hosted_selfcheck skipped: {exc}")
        return None


def add_hosted_summary(niche, ds, hosted_answers) -> None:
    """summary.json["hosted"]: Ultramarine rates with CIs, tokens, cost, the effort ablation and the
    hosted self-check — reported next to, never inside, the local open-model numbers."""
    lab_path, gens_path = config.DATA / "labelled_hosted.parquet", config.DATA / "generations_hosted.parquet"
    if not (lab_path.exists() and gens_path.exists()):
        return
    lab_h, gens_h = pd.read_parquet(lab_path), pd.read_parquet(gens_path)
    rates_h = rate_rows(lab_h, gens_h, niche)
    rates_h = rates_h[rates_h["scope"] == "model"].assign(scope="hosted")
    rates_h.to_csv(RATES_HOSTED_PATH, index=False)
    label_summary = json.loads((config.RESULTS / "label_summary.json").read_text(encoding="utf-8"))
    section = dict(label_summary.get("hosted", {}))
    for row in rates_h.to_dict("records"):
        section.setdefault("models", {}).setdefault(row["group"], {}).update(
            {k: row[k] for k in ("rate_lo", "rate_hi", "gen_share_lo", "gen_share_hi")})
    ledger = hosted.Ledger()
    section["spend"] = {"ledger_spent_usd": ledger.spent, "cap_usd": ledger.cap_usd,
                        "parts": ledger.data.get("parts", {})}
    if hosted_answers is not None:
        ghost_names = ds.loc[ds["y_ghost"], "name"].unique()
        real_names = ds.loc[~ds["y_ghost"], "name"].unique()
        section["selfcheck"] = {
            "model": hosted.MODEL, "n_names": int(ds["name"].nunique()),
            "ghost_names": int(len(ghost_names)),
            "ghost_names_said_exists": int(sum(hosted_answers[n]["answer"] == "yes" for n in ghost_names)),
            "real_names": int(len(real_names)),
            "real_names_said_missing": int(sum(hosted_answers[n]["answer"] == "no" for n in real_names)),
            "p_no_available": int(sum(hosted_answers[n].get("p_no") is not None for n in ds["name"].unique()))}
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))  # reload right before saving
    summary["hosted"] = section
    runconfig.save(summary, config.SUMMARY_PATH)


def write_summary(ds, y, unseen, eval_df, loo_df, timing_df, rates, threshold, answers, answers_prob, selfcheck_s,
                  final_fit_s, diff_rows, tabpfn_info) -> None:
    """Merge every headline number into results/summary.json (other stages add their own sections)."""
    label_summary = json.loads((config.RESULTS / "label_summary.json").read_text(encoding="utf-8"))
    summon = runconfig.load()["summon"]
    records = generate.in_plan(generate.read_records(config.RAW_GENERATIONS_PATH), summon)
    hours = {m: sum(r["latency_s"] for r in records if r["model"] == m) / 3600 for m in summon["models"]}

    def rate(scope, group):
        r = rates[(rates["scope"] == scope) & (rates["group"] == group)].iloc[0]
        return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in r.items()
                if k not in ("scope", "group")}

    wide = eval_df.pivot_table(index=["split", "arm"], columns="metric", values="value")
    ghost_names = ds.loc[ds["y_ghost"], "name"].unique()
    said_yes = sum(answers[n]["answer"] == "yes" for n in ghost_names)
    real_names = ds.loc[~ds["y_ghost"], "name"].unique()
    real_said_no = sum(answers[n]["answer"] == "no" for n in real_names)
    tab_time = timing_df[timing_df["arm"] == "tabpfn"]
    summary = runconfig.load(config.SUMMARY_PATH) if config.SUMMARY_PATH.exists() else {}
    summary.update({
        "updated_at": runconfig.utc_now(),
        "study": {
            "models": summon["models"], "n_models": len(summon["models"]), "n_prompts": summon["n_prompts"],
            "n_samples": len(summon["samples"]), "n_generations": label_summary["n_generations"],
            "n_generations_planned": label_summary["n_generations_planned"],
            "n_generations_aborted": label_summary["n_generations_aborted"],
            "generation_hours_by_model": hours, "generation_hours_total": sum(hours.values()),
            "temperature": summon["temperature"], "num_predict": summon["num_predict"],
        },
        "rates": {
            "overall": rate("overall", "all"),
            "per_model": {m: rate("model", m) for m in summon["models"]},
            "per_niche_level": {f"L{lv}": rate("niche_level", f"L{lv}") for lv in (1, 2, 3)},
            "l3_over_l1": rate("niche_level", "L3")["ghost_rate"] / rate("niche_level", "L1")["ghost_rate"],
            "ci": "95% bootstrap, 1,000 resamples by prompt",
        },
        "repeat": {
            "n_unique_ghost_names": label_summary["n_unique_ghost_names"],
            "repeated_2plus": label_summary["ghosts_repeated_2plus"],
            "repeated_2plus_share": label_summary["ghosts_repeated_2plus"] / label_summary["n_unique_ghost_names"],
            "multi_model": label_summary["ghosts_multi_model"],
            "multi_model_share": label_summary["ghosts_multi_model"] / label_summary["n_unique_ghost_names"],
        },
        "subtypes": label_summary["subtypes"],
        "born_after_summoning_mentions": label_summary["born_after_summoning_mentions"],
        "dataset": {"n_rows": int(len(ds)), "n_ghost": int(y.sum()), "ghost_share": float(y.mean()),
                    "n_unique_names": int(ds["name"].nunique()), "n_unseen_rows_split_c": int(unseen.sum()),
                    "n_unseen_ghost_rows_split_c": int(y[unseen].sum())},
        "eval": {split: {arm: {k: float(v) for k, v in wide.loc[(split, arm)].dropna().items()}
                         for arm in ARMS} for split in ("A", "B", "C")},
        "loo": {f"{r.held_out_model}|{r.arm}": {"auroc": r.auroc, "auroc_lo": r.auroc_lo, "auroc_hi": r.auroc_hi,
                                                 "auprc": r.auprc, "n": r.n, "n_ghost": r.n_ghost}
                for r in loo_df.itertuples()},
        "threshold": threshold,
        "auroc_diff_tabpfn_minus": {f"{r['split']}|{r['arm']}": {k: r[k] for k in ("diff", "lo", "hi", "share_a_better")}
                                    for r in diff_rows},
        "tabpfn": {**tabpfn_info, "subsampled": False, "max_train_rows": int(tab_time["n_train"].max()),
                   "ignore_pretraining_limits": False,
                   "fit_s_total": float(tab_time["fit_s"].sum()), "predict_s_total": float(tab_time["predict_s"].sum()),
                   "final_fit_s": final_fit_s, "n_fits": int(len(tab_time)) + 1},
        "timing_s": timing_df.groupby("arm")[["fit_s", "predict_s"]].sum().to_dict("index"),
        "selfcheck": {"model": selfcheck.MODEL, "prompt": selfcheck.PROMPT, "n_names": len(answers),
                      "answers": pd.Series([a["answer"] for a in answers.values()]).value_counts().to_dict(),
                      "ghost_names": int(len(ghost_names)), "ghost_names_said_exists": int(said_yes),
                      "real_names": int(len(real_names)), "real_names_said_missing": int(real_said_no),
                      "total_latency_s": selfcheck_s,
                      "prob": selfcheck_prob_summary(answers, answers_prob, ghost_names, real_names)},
    })
    runconfig.save(summary, config.SUMMARY_PATH)


def selfcheck_prob_summary(answers, answers_prob, ghost_names, real_names) -> dict:
    """How the probabilistic self-check was scored, and how it relates to the hard answers."""
    recs = list(answers_prob.values())
    p = lambda names: [answers_prob[n]["p_no"] for n in names if answers_prob[n].get("p_no") is not None]  # noqa: E731
    return {"score": "P(no) / (P(no) + P(yes)) over the first answer token's top-k alternatives",
            "top_k": selfcheck.TOP_K, "n_names": len(recs),
            "p_no_available": sum(r.get("p_no") is not None for r in recs),
            "fallback_to_hard_answer": sum(r.get("p_no") is None for r in recs),
            "hard_answer_same_as_first_ask": sum(answers_prob[n]["answer"] == answers[n]["answer"] for n in answers_prob),
            "median_p_no_ghost": float(np.median(p(ghost_names))) if p(ghost_names) else None,
            "median_p_no_real": float(np.median(p(real_names))) if p(real_names) else None,
            "total_latency_s": sum(r["latency_s"] for r in recs)}


def print_tables(eval_df, loo_df, timing_df, threshold) -> None:
    """Terminal summary: split x arm table, LOO AUROCs, timing, threshold."""
    wide = eval_df.pivot_table(index=["split", "arm"], columns="metric", values="value")
    for split, title in (("A", "Split A: by prompt"), ("B", "Split B: leave-one-model-out (pooled)"),
                         ("C", "Split C: unseen names")):
        table = Table(title=f"{title} (n={int(wide.loc[(split, ARMS[0]), 'n'])}, "
                            f"ghost={int(wide.loc[(split, ARMS[0]), 'n_ghost'])})")
        for col in ("arm", "AUROC [95% CI]", "AUPRC", "P@R90", "R@P95", "Brier", "ECE"):
            table.add_column(col, justify="left" if col == "arm" else "right")
        for arm in ARMS:
            r = wide.loc[(split, arm)]
            f = lambda v: "-" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
            table.add_row(arm, f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]", f(r.auprc),
                          f(r.precision_at_recall90), f(r.recall_at_precision95), f(r.brier), f(r.ece))
        console.print(table)
    console.print(loo_df.pivot_table(index="arm", columns="held_out_model", values="auroc").round(3).to_string())
    console.print(timing_df.groupby(["split", "arm"])[["fit_s", "predict_s"]].sum().round(1).to_string())
    console.print(f"threshold: tau={threshold['tau']:.4f} recall={threshold['recall']:.3f} "
                  f"precision={threshold['precision']:.3f} flagged {threshold['n_flagged']}/{threshold['n']}")


if __name__ == "__main__":
    main()
