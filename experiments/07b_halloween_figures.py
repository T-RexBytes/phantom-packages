"""Halloween "costume" versions of the result figures for the blog, saved to output/.

Each figure wears a different costume, and the decoration never carries data: every bar,
line and point is drawn from the same committed results as experiments/07_figures.py.
  1. graveyard        ghost rate by model: tombstones under a full moon
  2. candle ritual    ghost rate by niche: candles that burn higher on obscure tasks
  3. spider web       calibration: the diagonal is a web thread, TabPFN's bins are spiders
  4. ouija board      precision-recall: the yes/no baselines sit where the planchette stops
  5. pumpkin patch    repeat histogram: a parade of ghosts over jack-o'-lanterns
Inputs: results/*.csv, results/summary.json, data/oof_predictions.parquet (aggregated only).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from phantom import config  # noqa: E402

OUT = config.ROOT / "output"
W, H, DPI = 16, 10, 120  # 1920 x 1200 px
MODELS = ["gemma4:e4b", "phi4-mini:3.8b", "qwen3:4b-instruct"]
MODEL_COLORS = {"gemma4:e4b": "#e0703f", "phi4-mini:3.8b": "#a79cf0", "qwen3:4b-instruct": "#3fbf8f"}
FONTS = {k: f"C:/Windows/Fonts/{v}" for k, v in {
    "chiller": "CHILLER.TTF", "onyx": "ONYX.TTF", "viner": "VINERITC.TTF",
    "georgia": "georgiab.ttf", "jokerman": "JOKERMAN.TTF", "body": "segoeuib.ttf", "body_r": "segoeui.ttf"}.items()}
RNG_SEED = 31


def font(key: str, size: float) -> dict:
    """Font kwargs for a Windows font file, falling back to the default font."""
    try:
        return {"fontproperties": font_manager.FontProperties(fname=FONTS[key], size=size)}
    except (OSError, KeyError):
        return {"fontsize": size, "fontweight": "bold"}


def outline(width: float = 4, color: str = "#0b0710") -> list:
    return [pe.withStroke(linewidth=width, foreground=color)]


def canvas(top: str, bottom: str):
    """Figure + full-bleed background axes in pixel-like units (0..W*10, 0..H*10, equal aspect)."""
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.set_xlim(0, W * 10)
    bg.set_ylim(0, H * 10)
    bg.set_aspect("equal")
    bg.axis("off")
    grad = np.linspace(0, 1, 256)[:, None]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", [bottom, top])
    bg.imshow(grad, extent=[0, W * 10, 0, H * 10], origin="lower", cmap=cmap, aspect="auto", zorder=0)
    return fig, bg


def data_axes(fig, rect, ink: str, grid: str):
    ax = fig.add_axes(rect, zorder=2)
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink)
        ax.spines[s].set_alpha(0.6)
    ax.tick_params(colors=ink, labelsize=15)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font_manager.FontProperties(fname=FONTS["body"], size=15))
    ax.grid(True, color=grid, linewidth=1, alpha=0.55)
    ax.set_axisbelow(True)
    return ax


def style_ticks(ax, ink: str) -> None:
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font_manager.FontProperties(fname=FONTS["body"], size=15))
        lab.set_color(ink)
        lab.set_path_effects(outline(3))


def title(fig, text, sub, key, color, sub_color, size=64, sub_y=0.835):
    fig.text(0.05, 0.93, text, color=color, va="top", path_effects=outline(6), **font(key, size))
    fig.text(0.05, sub_y, sub, color=sub_color, va="top", path_effects=outline(4), **font("body_r", 17))


def footer(fig, color):
    fig.text(0.985, 0.015, "Phantom Packages · data: results/summary.json", ha="right", va="bottom",
             color=color, alpha=0.85, path_effects=outline(3), **font("body_r", 12))


def stars(bg, n=140, ymin=55, color="#fff6e0"):
    rng = np.random.default_rng(RNG_SEED)
    x, y = rng.uniform(0, W * 10, n), rng.uniform(ymin, H * 10, n)
    bg.scatter(x, y, s=rng.uniform(0.5, 5, n), color=color, alpha=0.7, linewidths=0, zorder=1)


def moon(bg, x, y, r, color="#f4ecd0", glow="#f4ecd0"):
    for k, a in ((2.4, 0.04), (1.8, 0.07), (1.35, 0.12)):
        bg.add_patch(Circle((x, y), r * k, color=glow, alpha=a, zorder=1, linewidth=0))
    bg.add_patch(Circle((x, y), r, color=color, zorder=1, linewidth=0))
    rng = np.random.default_rng(3)
    for _ in range(6):  # craters
        cx, cy = x + rng.uniform(-0.55, 0.55) * r, y + rng.uniform(-0.55, 0.55) * r
        bg.add_patch(Circle((cx, cy), r * rng.uniform(0.07, 0.17), color="#d9cfae", alpha=0.7, zorder=1, lw=0))


def bat(bg, x, y, s, color="#0b0710", z=1):
    pts = np.array([(0, 0), (0.35, 0.18), (0.6, 0.5), (0.75, 0.3), (1.0, 0.42), (1.2, 0.15), (1.5, 0.35),
                    (1.4, 0.0), (1.15, -0.15), (0.9, -0.05), (0.6, -0.25), (0.3, -0.1)])
    left = pts.copy()
    left[:, 0] = -left[:, 0]
    shape = np.vstack([pts, left[::-1]]) * s + (x, y)
    bg.add_patch(Polygon(shape, closed=True, color=color, zorder=z, linewidth=0))
    bg.add_patch(Ellipse((x, y + 0.05 * s), 0.35 * s, 0.5 * s, color=color, zorder=z, linewidth=0))


def bats(bg, spots, color="#0b0710"):
    for x, y, s in spots:
        bat(bg, x, y, s, color)


def tombstone_path(x0, x1, y0, y1, cap):
    """Rectangle with a semi-elliptical top of height `cap` (data coordinates)."""
    cx, rx = (x0 + x1) / 2, (x1 - x0) / 2
    t = np.linspace(np.pi, 0, 40)
    top = np.c_[cx + rx * np.cos(t), (y1 - cap) + cap * np.sin(t)]
    verts = np.vstack([[x0, y0], [x0, y1 - cap], top, [x1, y0], [x0, y0]])
    return MPath(verts, closed=True)


def ci_whisker(ax, x, lo, hi, color, w=0.08):
    ax.plot([x, x], [lo, hi], color=color, linewidth=2.4, zorder=6, solid_capstyle="round", path_effects=outline(5))
    for v in (lo, hi):
        ax.plot([x - w, x + w], [v, v], color=color, linewidth=2.4, zorder=6, path_effects=outline(5))


# ---------------------------------------------------------------- 1. graveyard

def graveyard(rates: pd.DataFrame, summary: dict) -> None:
    fig, bg = canvas("#0d2233", "#04090f")
    stars(bg)
    moon(bg, 136, 86, 9)
    bats(bg, [(110, 88, 4.2), (121, 82, 2.8), (100, 80, 2.2), (150, 63, 2.6)])
    hills = np.c_[np.linspace(0, W * 10, 60), 14 + 3 * np.sin(np.linspace(0, 7, 60))]
    bg.add_patch(Polygon(np.vstack([[0, 0], hills, [W * 10, 0]]), color="#081018", zorder=1))
    rng = np.random.default_rng(7)
    for x in np.linspace(2, 158, 70):  # grass tufts
        bg.plot([x, x + rng.uniform(-0.8, 0.8)], [14 + 3 * np.sin(x / 160 * 7), 16.5 + 3 * np.sin(x / 160 * 7)],
                color="#16301f", linewidth=2, zorder=1)
    for x, h in ((12, 8), (150, 9)):  # far-off crosses
        bg.plot([x, x], [15, 15 + h], color="#0a1720", linewidth=5, zorder=1)
        bg.plot([x - 2.4, x + 2.4], [15 + h * 0.7] * 2, color="#0a1720", linewidth=5, zorder=1)

    r = rates[rates["scope"] == "model"].set_index("group").loc[MODELS]
    allr = rates[rates["scope"] == "overall"].iloc[0]
    vals = list(r["ghost_rate"]) + [allr.ghost_rate]
    lo = list(r["rate_lo"]) + [allr.rate_lo]
    hi = list(r["rate_hi"]) + [allr.rate_hi]
    counts = [f"{int(a.ghosts)} of {int(a.mentions)}" for a in r.itertuples()] + \
             [f"{int(allr.ghosts)} of {int(allr.mentions):,}"]
    names = MODELS + ["all models"]
    glows = [MODEL_COLORS[m] for m in MODELS] + ["#cfc8d8"]

    ax = data_axes(fig, [0.09, 0.2, 0.86, 0.56], "#d9d4e4", "#2a3e52")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 0.30)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks(range(4), [f"{n}\n{c}" for n, c in zip(names, counts)])
    style_ticks(ax, "#e9e4f2")
    ax.tick_params(axis="x", length=0, pad=12)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("install suggestions that don't exist on PyPI", color="#d9d4e4", **font("body_r", 15))
    for i, (v, l, h, g) in enumerate(zip(vals, lo, hi, glows)):
        cap = min(0.035, v * 0.45)
        ax.add_patch(Ellipse((i, 0.0), 0.95, 0.018, color=g, alpha=0.35, zorder=3, linewidth=0))
        ax.add_patch(PathPatch(tombstone_path(i - 0.3, i + 0.3, 0, v, cap), facecolor="#8f8a96",
                               edgecolor="#2b2730", linewidth=2.5, zorder=4))
        ax.add_patch(PathPatch(tombstone_path(i - 0.3, i - 0.22, 0, v, cap * 0.6), facecolor="#a7a2ad",
                               edgecolor="none", zorder=4, alpha=0.6))
        ax.plot([i - 0.3, i + 0.3], [0.004, 0.004], color=g, linewidth=6, zorder=5, solid_capstyle="butt")
        if v > 0.06:
            ax.text(i, min(l - 0.012, v * 0.55), "R.I.P.", ha="center", va="top", color="#4a4450", zorder=7,
                    **font("georgia", 18))
        ci_whisker(ax, i, l, h, "#f4ecd0")
        ax.text(i, h + 0.008, f"{v:.1%}", ha="center", va="bottom", color="#fff6e0",
                path_effects=outline(5), **font("body", 26))
    title(fig, "Where the ghosts are buried", f"Ghost rate by model · {summary['study']['n_generations']:,} "
          "generations of 200 coding tasks · whiskers: 95% bootstrap CI by prompt", "chiller", "#e8f0f4", "#c9d6df",
          size=78, sub_y=0.815)
    footer(fig, "#9fb2c0")
    fig.savefig(OUT / "halloween_ghost_rate_by_model.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 2. candle ritual

def candles(rates: pd.DataFrame) -> None:
    fig, bg = canvas("#2a0d1f", "#0a0309")
    rng = np.random.default_rng(11)
    for _ in range(40):  # embers
        bg.add_patch(Circle((rng.uniform(0, 160), rng.uniform(10, 100)), rng.uniform(0.15, 0.5),
                            color="#ffb347", alpha=rng.uniform(0.15, 0.5), zorder=1, lw=0))
    ring = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 2  # faint pentagram circle behind
    pts = np.c_[118 + 30 * np.cos(ring), 52 + 30 * np.sin(ring)]
    bg.add_patch(Circle((118, 52), 30, fill=False, edgecolor="#5c2040", linewidth=2, alpha=0.6, zorder=1))
    star = pts[[0, 2, 4, 1, 3, 0]]
    bg.plot(star[:, 0], star[:, 1], color="#5c2040", linewidth=2, alpha=0.6, zorder=1)
    bg.add_patch(Polygon([[0, 0], [160, 0], [160, 12], [0, 12]], color="#120509", zorder=1))  # altar

    r = rates[rates["scope"] == "niche_level"].set_index("group").loc[["L1", "L2", "L3"]]
    labels = {"L1": "L1 · everyday", "L2": "L2 · specialised", "L3": "L3 · obscure"}
    ax = data_axes(fig, [0.09, 0.17, 0.62, 0.6], "#ead7c8", "#4a2238")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 0.45)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks(range(3), [f"{labels[g]}\n{int(x.ghosts)} of {int(x.mentions)}" for g, x in r.iterrows()])
    style_ticks(ax, "#f6e6d6")
    ax.tick_params(axis="x", length=0, pad=12)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("install suggestions that don't exist on PyPI", color="#ead7c8", **font("body_r", 15))
    for i, (g, x) in enumerate(r.iterrows()):
        v = x.ghost_rate
        ax.add_patch(Polygon([[i - 0.24, 0], [i + 0.24, 0], [i + 0.24, v], [i - 0.24, v]], closed=True,
                             facecolor="#f3e2c0", edgecolor="#6b4a2a", linewidth=2, zorder=4))
        ax.add_patch(Polygon([[i - 0.24, 0], [i - 0.12, 0], [i - 0.12, v], [i - 0.24, v]], closed=True,
                             facecolor="#fff4dc", edgecolor="none", zorder=4, alpha=0.7))
        for dx, dl in ((-0.16, 0.25), (0.05, 0.4), (0.17, 0.18)):  # wax drips
            d = min(dl * v, 0.05)
            ax.add_patch(Ellipse((i + dx, v - d), 0.07, d * 2, color="#f3e2c0", zorder=5, linewidth=0))
        ax.plot([i, i], [v, v + 0.012], color="#2b1a10", linewidth=2.5, zorder=6)  # wick
        for k, (wd, ht, c, a) in enumerate(((0.2, 0.075, "#ff7a1a", 0.35), (0.12, 0.055, "#ffb43c", 0.9),
                                            (0.06, 0.03, "#fff3b0", 1.0))):
            ax.add_patch(Ellipse((i, v + 0.012 + ht / 2), wd, ht, color=c, alpha=a, zorder=7 + k, linewidth=0))
        ax.add_patch(Ellipse((i, 0), 0.75, 0.02, color="#7a4a1e", zorder=3, linewidth=0))
        ci_whisker(ax, i + 0.36, x.rate_lo, x.rate_hi, "#ffd18a", w=0.05)
        ax.text(i, v + 0.1, f"{v:.1%}", ha="center", va="bottom", color="#fff2dc",
                path_effects=outline(5), **font("body", 28))
    l1, l3 = r.loc["L1", "ghost_rate"], r.loc["L3", "ghost_rate"]
    fig.text(0.76, 0.6, f"On obscure tasks\nthe models invented\n{l3 / l1:.0f}× as many\npackages as on\neveryday ones.",
             color="#ffd18a", va="top", path_effects=outline(5), **font("onyx", 40))
    title(fig, "The darker the task, the brighter they burn", "Ghost rate by how niche the task was · all three "
          "models pooled · gold whiskers: 95% CI", "onyx", "#ffe0b0", "#e8cdb8", size=70)
    footer(fig, "#c9a99a")
    fig.savefig(OUT / "halloween_ghost_rate_by_niche.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 3. spider web

def spider(ax, x, y, s, color, thread_top=1.02):
    ax.plot([x, x], [y, thread_top], color="#d8d8e0", linewidth=0.8, alpha=0.5, zorder=4)
    for side in (-1, 1):
        for k, ang in enumerate((0.5, 0.15, -0.2, -0.55)):
            dx, dy = side * np.cos(ang) * s * 2.2, np.sin(ang) * s * 2.2
            ax.plot([x, x + dx * 0.55, x + dx], [y, y + dy * 0.9 + s * 0.6, y + dy - s * 0.4], color="#e9dcc8",
                    linewidth=1.8, zorder=6, solid_capstyle="round")
    ax.add_patch(Ellipse((x, y), s * 1.4, s * 1.7, color=color, zorder=7, ec="#0b0710", lw=1.5))
    ax.add_patch(Circle((x, y + s * 1.05), s * 0.55, color=color, zorder=7, ec="#0b0710", lw=1.5))


def web(calib: pd.DataFrame) -> None:
    fig, bg = canvas("#101018", "#050508")
    moon(bg, 140, 82, 11, color="#e9e9f2", glow="#c9d0ff")
    cx, cy = 0, 100  # web anchored in the top-left corner
    angles = np.linspace(-np.pi / 2, 0, 9)
    for a in angles:
        bg.plot([cx, cx + 120 * np.cos(a)], [cy, cy + 120 * np.sin(a)], color="#c8c8d4", alpha=0.18, lw=1.2, zorder=1)
    for rad in np.arange(10, 120, 11):
        pts = [(cx + rad * np.cos(a) * (0.96 + 0.04 * np.cos(4 * a)), cy + rad * np.sin(a)) for a in angles]
        xs, ys = zip(*pts)
        bg.plot(xs, ys, color="#c8c8d4", alpha=0.14, lw=1, zorder=1)
    bats(bg, [(118, 90, 1.6), (155, 70, 1.2)], color="#000000")

    ax = data_axes(fig, [0.1, 0.12, 0.5, 0.64], "#dcdce6", "#2a2a38")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style_ticks(ax, "#ececf4")
    ax.plot([0, 1], [0, 1], color="#e8e8f0", linewidth=1.6, alpha=0.8, zorder=3, linestyle=(0, (1.5, 3)))
    ax.text(0.98, 0.06, "dotted line: perfect calibration", ha="right", color="#e8e8f0", path_effects=outline(4),
            **font("body_r", 14))
    gb = calib[(calib["arm"] == "hgb") & (calib["n"] >= 5)]
    ax.plot(gb["mean_pred"], gb["frac_ghost"], color="#8f8fa8", linewidth=2, zorder=4, marker="o", ms=6)
    t = calib[(calib["arm"] == "tabpfn") & (calib["n"] >= 5)]
    ax.plot(t["mean_pred"], t["frac_ghost"], color="#ff8a2a", linewidth=4, zorder=5, path_effects=outline(7, "#3a1a05"))
    for row in t.itertuples():
        spider(ax, row.mean_pred, row.frac_ghost, 0.016 + 0.00006 * min(row.n, 200), "#ff8a2a")
    ax.set_xlabel("TabPFN's predicted P(ghost), 10 equal-width bins", color="#dcdce6", **font("body_r", 15))
    ax.set_ylabel("share that really were ghosts", color="#dcdce6", **font("body_r", 15))
    ece = {a: float((c["n"] * (c["mean_pred"] - c["frac_ghost"]).abs()).sum() / c["n"].sum())
           for a, c in calib[calib["n"] > 0].groupby("arm")}
    fig.text(0.65, 0.7, "Each spider is one bin of\nTabPFN predictions. On the\nthread means honest.",
             color="#f0f0f8", va="top", path_effects=outline(4), **font("viner", 26))
    fig.text(0.65, 0.42, f"calibration error (ECE)\nTabPFN  {ece['tabpfn']:.3f}\ngradient boosting  "
             f"{ece['hgb']:.3f}\nlogistic regression  {ece['logreg']:.3f}", color="#ffb46a", va="top",
             linespacing=1.6, path_effects=outline(4), **font("body", 20))
    fig.text(0.65, 0.18, "grey line: gradient boosting\nbins with < 5 rows hidden", color="#b8b8cc", va="top",
             path_effects=outline(3), **font("body_r", 14))
    title(fig, "Caught in the web of honest odds", "When TabPFN says 80%, about 80% really are ghosts · "
          "out-of-fold, split A (by prompt)", "viner", "#f2f2fa", "#c8c8d8", size=56, sub_y=0.815)
    footer(fig, "#9a9ab0")
    fig.savefig(OUT / "halloween_calibration.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 4. ouija board

def planchette(ax, x, y, s, color="#f3e2c0"):
    t = np.linspace(0, 2 * np.pi, 80)
    heart = np.c_[x + s * 1.0 * np.sin(t) * (1 - 0.25 * np.cos(t)), y + s * 1.25 * (0.85 * np.cos(t) - 0.15)]
    for patch in (Polygon(heart, closed=True, facecolor=color, edgecolor="#2b1a10", lw=2.5, zorder=8, alpha=0.92),
                  Circle((x, y), s * 0.42, facecolor="#3b2414", edgecolor="#2b1a10", lw=2, zorder=9),
                  Circle((x, y), s * 0.32, facecolor="none", edgecolor="#f3e2c0", lw=1.5, zorder=10)):
        patch.set_clip_on(False)  # points at recall 1.0 sit on the axis edge
        ax.add_patch(patch)


def ouija(oof: pd.DataFrame, summary: dict) -> None:
    fig, bg = canvas("#5a3720", "#24140a")
    rng = np.random.default_rng(5)
    for y in np.linspace(2, 98, 48):  # wood grain
        xs = np.linspace(0, 160, 200)
        bg.plot(xs, y + 0.6 * np.sin(xs / rng.uniform(6, 14) + rng.uniform(0, 6)), color="#2f1b0d",
                alpha=0.25, lw=1.2, zorder=1)
    bg.add_patch(Ellipse((80, 50), 190, 150, facecolor="none", edgecolor="#1a0d05", lw=120, alpha=0.35, zorder=1))
    for k, ch in enumerate("1234567890"):  # the board's number row, faint
        bg.text(118 + k * 4.0, 4.5, ch, color="#e8d2a8", alpha=0.28, ha="center", **font("georgia", 22))
    bg.text(8, 4.5, "GOOD BYE", color="#e8d2a8", alpha=0.28, **font("georgia", 26))
    for k, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):  # letter arc in the right panel
        a = np.pi * (0.92 - 0.84 * k / 25)
        bg.text(134 + 22 * np.cos(a), 80 + 11 * np.sin(a), ch, color="#e8d2a8", alpha=0.32, ha="center",
                va="center", rotation=np.degrees(a) - 90, **font("georgia", 14))

    y = oof["y_ghost"].to_numpy(int)
    ax = data_axes(fig, [0.09, 0.14, 0.6, 0.62], "#f0dcb4", "#6a4428")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    style_ticks(ax, "#f7e8c8")
    ax.axvline(0.9, color="#f0dcb4", linewidth=1.4, linestyle=(0, (4, 4)), alpha=0.8, zorder=3)
    ax.text(0.885, 0.04, "90% recall: the guard's target", rotation=90, ha="right", va="bottom", color="#f0dcb4",
            path_effects=outline(3, "#24140a"), **font("body_r", 13))
    styles = [("rule_editdist", "#b08a5e", 2.2, "edit-distance rule"), ("logreg", "#d2b07c", 2.2, "logistic regression"),
              ("hgb", "#f0d6a0", 2.2, "gradient boosting"), ("tabpfn", "#ff9b2e", 4.5, "TabPFN-3.5")]
    for k, (arm, color, lw, label) in enumerate(styles):
        p, r, _ = precision_recall_curve(y, oof[f"A_{arm}"])
        ax.plot(r, p, color=color, linewidth=lw, drawstyle="steps-post", zorder=5 if arm == "tabpfn" else 4,
                path_effects=outline(lw + 3, "#24140a"))
        ax.plot([0.05, 0.11], [0.33 - 0.065 * (3 - k)] * 2, color=color, linewidth=lw, transform=ax.transAxes)
        ax.text(0.13, 0.33 - 0.065 * (3 - k), f"{label}   AUPRC {summary['eval']['A'][arm]['auprc']:.2f}",
                transform=ax.transAxes, va="center", color="#fff1d6", path_effects=outline(4, "#24140a"),
                **font("body" if arm == "tabpfn" else "body_r", 15))
    for arm, label, at in (("llm_selfcheck", "gemma4 asks itself", (0.62, 0.5)),
                           ("rule_top15k", '"not in top 15k" rule', (0.62, 0.62))):
        flag = oof[f"A_{arm}"].to_numpy() >= 0.5
        tp = int((flag & (y == 1)).sum())
        rec, prec = tp / y.sum(), tp / max(int(flag.sum()), 1)
        planchette(ax, rec, prec, 0.03)  # the planchette's window sits on the point
        ax.annotate(f"{label} (yes/no)\nrecall {rec:.2f} · precision {prec:.2f}", (rec - 0.03, prec), xytext=at,
                    ha="right", va="center", color="#fff1d6", path_effects=outline(4, "#24140a"),
                    arrowprops={"arrowstyle": "-", "color": "#f0dcb4", "lw": 1.2, "alpha": 0.8},
                    **font("body_r", 14))
    ax.set_xlabel("recall (share of ghosts caught)", color="#f0dcb4", **font("body_r", 15))
    ax.set_ylabel("precision (share of flags that are ghosts)", color="#f0dcb4", **font("body_r", 15))
    fig.text(0.735, 0.74, "YES", color="#f3e2c0", path_effects=outline(5, "#24140a"), **font("georgia", 54))
    fig.text(0.735, 0.62, "…but only after\nPyPI has spoken.", color="#f3e2c0", va="top", linespacing=1.3,
             path_effects=outline(4, "#24140a"), **font("georgia", 24))
    sc = summary["selfcheck"]
    fig.text(0.735, 0.43, f"Asked “does it exist?”,\ngemma4 said NO to\n{sc['real_names_said_missing']} of "
             f"{sc['real_names']} real packages.", color="#ffcf8a", va="top", linespacing=1.35,
             path_effects=outline(4, "#24140a"), **font("georgia", 22))
    title(fig, "Ask the board: is it a ghost?", "Precision vs. recall before asking PyPI · out-of-fold, split A "
          "(GroupKFold by prompt) · 888 names, 173 ghosts", "georgia", "#fbe9c6", "#ead2a8", size=52)
    footer(fig, "#d8bf96")
    fig.savefig(OUT / "halloween_precision_recall.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 5. pumpkin patch

def ghost_bar(ax, x, h, w, top_color, low_h=0.0, low_color="#ff7a1a"):
    """A ghost whose body height is the bar value; the lower `low_h` is shaded as a second segment."""
    if h <= 0:
        return
    cap = min(h * 0.45, 9)
    t = np.linspace(np.pi, 0, 40)
    top = np.c_[x + w / 2 * np.cos(t), (h - cap) + cap * np.sin(t)]
    hem = np.c_[np.linspace(x + w / 2, x - w / 2, 25), -0.6 + 0.9 * np.abs(np.sin(np.linspace(0, 3 * np.pi, 25)))]
    body = np.vstack([[x - w / 2, h - cap], top, [x + w / 2, 0.3], hem])
    ax.add_patch(Polygon(body, closed=True, facecolor=top_color, edgecolor="#2a1206", lw=2, zorder=4))
    if low_h > 0:
        seg = np.vstack([[x - w / 2, min(low_h, h - cap)], [x + w / 2, min(low_h, h - cap)], [x + w / 2, 0.3], hem])
        ax.add_patch(Polygon(seg, closed=True, facecolor=low_color, edgecolor="#2a1206", lw=2, zorder=5))
    if h > 6:
        ey = h - cap * 0.75
        for dx in (-w * 0.18, w * 0.18):
            ax.add_patch(Ellipse((x + dx, ey), w * 0.13, min(cap * 0.3, 3.2), color="#1a0c05", zorder=6))
        ax.add_patch(Ellipse((x, ey - min(cap * 0.32, 3.5)), w * 0.12, min(cap * 0.22, 2.6), color="#1a0c05", zorder=6))


def pumpkin(bg, x, y, s):
    for dx, c in ((-0.55, "#c4500f"), (0.55, "#c4500f"), (-0.25, "#e8661a"), (0.25, "#e8661a"), (0, "#f07a22")):
        bg.add_patch(Ellipse((x + dx * s, y), s * 0.9, s * 1.15, color=c, zorder=2, lw=0))
    bg.add_patch(Polygon([[x - 0.1 * s, y + 0.5 * s], [x + 0.12 * s, y + 0.5 * s], [x + 0.2 * s, y + 0.85 * s],
                          [x, y + 0.8 * s]], color="#3b5a1f", zorder=2))
    glow = "#ffd23a"
    for dx in (-0.35, 0.35):
        bg.add_patch(Polygon([[x + dx * s - 0.15 * s, y + 0.05 * s], [x + dx * s + 0.15 * s, y + 0.05 * s],
                              [x + dx * s, y + 0.3 * s]], color=glow, zorder=3))
    mouth = [[x - 0.5 * s, y - 0.15 * s], [x - 0.3 * s, y - 0.3 * s], [x - 0.15 * s, y - 0.2 * s],
             [x, y - 0.35 * s], [x + 0.15 * s, y - 0.2 * s], [x + 0.3 * s, y - 0.3 * s], [x + 0.5 * s, y - 0.15 * s],
             [x + 0.3 * s, y - 0.42 * s], [x, y - 0.5 * s], [x - 0.3 * s, y - 0.42 * s]]
    bg.add_patch(Polygon(mouth, color=glow, zorder=3))


def pumpkin_patch(counts: pd.DataFrame, summary: dict) -> None:
    fig, bg = canvas("#2b1405", "#070302")
    stars(bg, n=80, ymin=70, color="#ffe2b0")
    moon(bg, 145, 84, 7, color="#ffd9a0", glow="#ff9a40")
    bats(bg, [(124, 90, 1.8), (132, 82, 1.2), (110, 93, 1.0)])
    bg.add_patch(Polygon([[0, 0], [160, 0], [160, 9], [0, 11]], color="#120804", zorder=1))
    for x, s in ((6, 4.2), (18, 3.2), (148, 4.6), (136, 3.0), (157, 2.6)):
        pumpkin(bg, x, 7 + s * 0.55, s)
    vines = np.linspace(0, 160, 120)
    bg.plot(vines, 9.5 + 0.8 * np.sin(vines / 4), color="#2f4a17", lw=2.2, zorder=1)

    by = counts.set_index("mentions")
    xs = by.index.to_numpy()
    total = by["ghost_names"].to_numpy()
    multi = by["by_2plus_models"].to_numpy()
    ax = data_axes(fig, [0.12, 0.2, 0.62, 0.56], "#f6dcc0", "#4a2410")
    ax.set_xlim(0.4, xs.max() + 0.6)
    ax.set_ylim(-2, total.max() * 1.18)
    style_ticks(ax, "#fde8d0")
    ax.set_xticks(xs, [f"{i}×" for i in xs])
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=10)
    for x, n, m in zip(xs, total, multi):
        ghost_bar(ax, x, n, 0.62, "#ff7a1a" if m == n else "#f4f1ea")
        if 0 < m < n:
            ax.add_patch(Polygon([[x - 0.31, 0.3], [x + 0.31, 0.3], [x + 0.31, m], [x - 0.31, m]], closed=True,
                                 facecolor="#ff7a1a", edgecolor="#2a1206", lw=2, zorder=5, alpha=0.95))
        ax.text(x, n + total.max() * 0.03, f"{n}", ha="center", va="bottom", color="#fff3e0",
                path_effects=outline(5), **font("body", 26))
    ax.set_xlabel("times the same ghost name was suggested", color="#f6dcc0", **font("body_r", 16))
    ax.set_ylabel("ghost names", color="#f6dcc0", **font("body_r", 16))
    rep = summary["repeat"]
    fig.text(0.77, 0.72, f"{rep['repeated_2plus']} of {rep['n_unique_ghost_names']}\nghosts\ncame back",
             color="#ffb347", va="top", path_effects=outline(5), **font("jokerman", 34))
    fig.text(0.77, 0.47, "white: suggested by\none model\norange: suggested by\ntwo or more models",
             color="#fde8d0", va="top", linespacing=1.4, path_effects=outline(4), **font("body", 17))
    fig.text(0.77, 0.27, "Every repeat came\nfrom the same task.\nA predictable ghost is\na name worth squatting.",
             color="#f6dcc0", va="top", linespacing=1.4, path_effects=outline(4), **font("body_r", 16))
    title(fig, "The ghosts that keep coming back", f"{rep['n_unique_ghost_names']} unregistered ghost names "
          "by how often a model suggested them · names withheld", "jokerman", "#ffcf7a", "#f0d2b4", size=52)
    footer(fig, "#c8a07a")
    fig.savefig(OUT / "halloween_repeat_ghosts.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()
    OUT.mkdir(exist_ok=True)
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    rates = pd.read_csv(config.RESULTS / "rates.csv")
    graveyard(rates, summary)
    candles(rates)
    web(pd.read_csv(config.RESULTS / "calibration.csv"))
    ouija(pd.read_parquet(config.DATA / "oof_predictions.parquet"), summary)
    pumpkin_patch(pd.read_csv(config.RESULTS / "repeat_counts.csv"), summary)
    for p in sorted(OUT.glob("halloween_*.png")):
        print(f"wrote {p.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
