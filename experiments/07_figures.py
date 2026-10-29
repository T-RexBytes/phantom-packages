"""Stage 07: figures (BUILD_SPEC §5.7). Dark Halloween theme, readable first.

Writes results/{rates_by_model,rates_by_niche,calibration,pr_curves,repeat_hist}.png and
results/repeat_counts.csv. Inputs are committed results plus private per-row predictions and
ghost counts, but every output is an aggregate: no package names are drawn or written.

Palette (validated with the dataviz validator, dark mode, surface #15121c, all pairs):
pumpkin #d95926, violet #9085e9, aqua #199e70 for the three models; marigold #e8a33d is the
accent for highlights; greys for baselines.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from phantom import config  # noqa: E402

BG = "#15121c"
INK, INK2, MUTED = "#f0ece4", "#c9c2d3", "#a39bb0"
GRID = "#2e2838"
ACCENT = "#e8a33d"
MODEL_COLORS = {"gemma4:e4b": "#d95926", "phi4-mini:3.8b": "#9085e9", "qwen3:4b-instruct": "#199e70"}
BASELINE_GREYS = ["#b8b2c4", "#8a8497", "#6c6679"]
ARM_LABELS = {"rule_top15k": '"not in top 15k" rule', "rule_editdist": "edit-distance rule",
              "logreg": "logistic regression", "hgb": "gradient boosting", "tabpfn": "TabPFN-3.5",
              "llm_selfcheck": "gemma4 asks itself"}
FIGS = {name: config.RESULTS / f"{name}.png"
        for name in ("rates_by_model", "rates_by_niche", "calibration", "pr_curves", "repeat_hist")}
REPEAT_CSV = config.RESULTS / "repeat_counts.csv"
MIN_BIN_ROWS = 5  # calibration bins with fewer rows are noise, not signal
FOOT = "Phantom Packages · data: results/"


def style() -> None:
    """Global matplotlib style: dark surface, recessive grid, light ink."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 12, "legend.frameon": False, "legend.labelcolor": INK2,
    })


def finish(fig, ax, path, subtitle: str) -> None:
    """Subtitle under the title, source footer under the axes, save at 150 dpi."""
    lines = subtitle.count("\n") + 1
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=14 + 16 * lines)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=INK2, fontsize=11, va="bottom")
    fig.text(0.99, -0.03, FOOT, ha="right", va="top", color=MUTED, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def bar_with_ci(ax, labels, rates, lo, hi, colors, ylabel_fmt="{:.1%}") -> None:
    """Vertical bars with 95% CI whiskers and a direct value label above each whisker."""
    x = np.arange(len(labels))
    ax.bar(x, rates, width=0.56, color=colors, edgecolor=BG, linewidth=2, zorder=2)
    ax.errorbar(x, rates, yerr=[np.array(rates) - lo, np.array(hi) - rates], fmt="none",
                ecolor=INK2, elinewidth=1.6, capsize=6, zorder=3)
    for xi, r, h in zip(x, rates, hi):
        ax.text(xi, h + 0.012, ylabel_fmt.format(r), ha="center", va="bottom", color=INK, fontsize=12,
                fontweight="bold")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(hi) * 1.22)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", colors=INK2, labelsize=12)


def fig_rates_by_model(rates: pd.DataFrame, summary: dict) -> None:
    r = rates[rates["scope"] == "model"].set_index("group").loc[list(MODEL_COLORS)]
    allr = rates[rates["scope"] == "overall"].iloc[0]
    labels = [f"{m}\n{int(row.ghosts)} of {int(row.mentions)}" for m, row in r.iterrows()] + \
             [f"all models\n{int(allr.ghosts)} of {int(allr.mentions):,}"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bar_with_ci(ax, labels, list(r["ghost_rate"]) + [allr.ghost_rate], np.r_[r["rate_lo"], allr.rate_lo],
                np.r_[r["rate_hi"], allr.rate_hi], [*MODEL_COLORS.values(), BASELINE_GREYS[1]])
    ax.set_title("Ghost rate by model")
    ax.set_ylabel("install suggestions that don't exist on PyPI")
    finish(fig, ax, FIGS["rates_by_model"],
           f"{summary['study']['n_generations']:,} generations, 200 prompts · whiskers: 95% bootstrap CI by prompt")


def fig_rates_by_niche(rates: pd.DataFrame) -> None:
    r = rates[rates["scope"] == "niche_level"].set_index("group").loc[["L1", "L2", "L3"]]
    names = {"L1": "L1 everyday", "L2": "L2 specialised", "L3": "L3 obscure"}
    labels = [f"{names[g]}\n{int(row.ghosts)} of {int(row.mentions)}" for g, row in r.iterrows()]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    bar_with_ci(ax, labels, r["ghost_rate"].to_numpy(), r["rate_lo"].to_numpy(), r["rate_hi"].to_numpy(),
                [ACCENT] * 3)
    ax.set_title("The more niche the task, the more ghosts")
    ax.set_ylabel("install suggestions that don't exist on PyPI")
    l1, l3 = r.loc["L1", "ghost_rate"], r.loc["L3", "ghost_rate"]
    finish(fig, ax, FIGS["rates_by_niche"], f"all 3 models pooled · L3 is {l3 / l1:.0f}x L1 · whiskers: 95% CI")


def fig_calibration(calib: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 4)), zorder=1)
    ax.text(0.30, 0.25, "perfect calibration", rotation=42, color=MUTED, fontsize=10, ha="center")
    series = [("hgb", BASELINE_GREYS[1], 1.6), ("tabpfn", ACCENT, 2.6)]
    for arm, color, lw in series:
        c = calib[(calib["arm"] == arm) & (calib["n"] >= MIN_BIN_ROWS)]
        ax.plot(c["mean_pred"], c["frac_ghost"], color=color, linewidth=lw, zorder=3 if arm == "tabpfn" else 2)
        ax.scatter(c["mean_pred"], c["frac_ghost"], s=np.clip(c["n"], 12, 400) * 0.6 + 30, color=color,
                   edgecolor=BG, linewidth=2, zorder=4 if arm == "tabpfn" else 2)
        last = c.iloc[-1]
        ax.annotate(ARM_LABELS[arm], (last["mean_pred"], last["frac_ghost"]),
                    xytext=(-14, 14) if arm == "tabpfn" else (-6, -34), textcoords="offset points",
                    color=INK if arm == "tabpfn" else INK2, fontsize=11,
                    fontweight="bold" if arm == "tabpfn" else "normal", ha="right", annotation_clip=False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("predicted P(ghost), 10 equal-width bins")
    ax.set_ylabel("share that really were ghosts")
    ax.set_title("When TabPFN says 80%, it means about 80%")
    ece = {a: calib_ece(calib[calib["arm"] == a]) for a in ("tabpfn", "hgb", "logreg")}
    finish(fig, ax, FIGS["calibration"],
           f"out-of-fold, split A · dot size = rows in bin · bins with < {MIN_BIN_ROWS} rows hidden\n"
           f"ECE (all bins): TabPFN {ece['tabpfn']:.3f} · gradient boosting {ece['hgb']:.3f} · "
           f"logistic regression {ece['logreg']:.3f}")


def calib_ece(c: pd.DataFrame) -> float:
    c = c[c["n"] > 0]
    return float((c["n"] * (c["mean_pred"] - c["frac_ghost"]).abs()).sum() / c["n"].sum())


def fig_pr_curves(oof: pd.DataFrame, summary: dict) -> None:
    y = oof["y_ghost"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.axvline(0.9, color=MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.text(0.9, 0.06, " 90% recall\n (guard target)", color=MUTED, fontsize=10, va="bottom")
    curves = [("rule_editdist", BASELINE_GREYS[2], 1.6), ("logreg", BASELINE_GREYS[1], 1.6),
              ("hgb", BASELINE_GREYS[0], 1.6), ("tabpfn", ACCENT, 2.8)]
    label_y = {"tabpfn": 0.40, "hgb": 0.33, "logreg": 0.27, "rule_editdist": 0.21}
    for arm, color, lw in curves:
        p, r, _ = precision_recall_curve(y, oof[f"A_{arm}"])
        ax.plot(r, p, color=color, linewidth=lw, zorder=3 if arm == "tabpfn" else 2, drawstyle="steps-post")
        auprc = summary["eval"]["A"][arm]["auprc"]
        ax.plot([0.04, 0.09], [label_y[arm] + 0.012] * 2, color=color, linewidth=lw, transform=ax.transAxes)
        ax.text(0.11, label_y[arm], f"{ARM_LABELS[arm]}  AUPRC {auprc:.2f}", color=INK if arm == "tabpfn" else INK2,
                fontsize=11, fontweight="bold" if arm == "tabpfn" else "normal", transform=ax.transAxes)
    for arm, marker in (("rule_top15k", "s"), ("llm_selfcheck", "D")):
        s = oof[f"A_{arm}"].to_numpy() >= 0.5
        tp = int((s & (y == 1)).sum())
        rec, prec = tp / y.sum(), tp / max(int(s.sum()), 1)
        ax.scatter([rec], [prec], marker=marker, s=90, color=BASELINE_GREYS[0], edgecolor=BG, linewidth=2, zorder=5)
        ax.annotate(f"{ARM_LABELS[arm]} (yes/no)", (rec, prec), xytext=(-10, -18 if arm == "rule_top15k" else 10),
                    textcoords="offset points", ha="right", color=INK2, fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall (share of ghosts caught)")
    ax.set_ylabel("precision (share of flags that are ghosts)")
    ax.set_title("Spotting a ghost before asking PyPI")
    finish(fig, ax, FIGS["pr_curves"], "out-of-fold predictions, split A (GroupKFold by prompt), 888 rows, 173 ghosts")


def fig_repeat_hist(counts: pd.DataFrame) -> None:
    by = counts.groupby("mentions").agg(names=("mentions", "size"), multi_model=("n_models", lambda s: int((s >= 2).sum())))
    by = by.reindex(range(1, int(by.index.max()) + 1), fill_value=0)
    by["single_model"] = by["names"] - by["multi_model"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = by.index.to_numpy()
    ax.bar(x, by["single_model"], width=0.6, color=ACCENT, edgecolor=BG, linewidth=2, label="one model", zorder=2)
    ax.bar(x, by["multi_model"], width=0.6, bottom=by["single_model"], color="#d95926", edgecolor=BG, linewidth=2,
           label="two or more models", zorder=2)
    for xi, n in zip(x, by["names"]):
        ax.text(xi, n + max(by["names"]) * 0.015, f"{n}", ha="center", va="bottom", color=INK, fontsize=12,
                fontweight="bold")
    ax.set_ylim(0, max(by["names"]) * 1.15)
    ax.set_xticks(x, [f"{i}x" for i in x])
    ax.set_xlabel("times a ghost name was suggested")
    ax.set_ylabel("ghost names")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Most ghosts appear once. Some keep coming back")
    repeat = int((counts["mentions"] >= 2).sum())
    finish(fig, ax, FIGS["repeat_hist"],
           f"{len(counts)} unregistered ghost names · {repeat} ({repeat / len(counts):.0%}) suggested 2+ times · "
           "every repeat came from the same prompt")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()  # accepts --fast
    style()
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    rates = pd.read_csv(config.RESULTS / "rates.csv")
    fig_rates_by_model(rates, summary)
    fig_rates_by_niche(rates)
    fig_calibration(pd.read_csv(config.RESULTS / "calibration.csv"))
    fig_pr_curves(pd.read_parquet(config.DATA / "oof_predictions.parquet"), summary)

    ghosts = pd.read_csv(config.PRIVATE / "ghost_counts.csv")  # private input, aggregate output
    counts = ghosts[["mentions", "models"]].rename(columns={"models": "n_models"})
    agg = counts.groupby("mentions").agg(ghost_names=("mentions", "size"),
                                         by_2plus_models=("n_models", lambda s: int((s >= 2).sum())))
    agg.to_csv(REPEAT_CSV)
    fig_repeat_hist(counts)
    for path in FIGS.values():
        print(f"wrote {path.relative_to(config.ROOT)}")
    print(f"wrote {REPEAT_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
