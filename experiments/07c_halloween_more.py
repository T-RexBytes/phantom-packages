"""More Halloween figures for the blog, saved to output/: the findings after the first five.

Same costume rules as experiments/07b_halloween_figures.py, whose primitives this reuses:
  6. haunted mirror   run 2 vs run 1: the same ghost names came back
  7. potion shelf     Ultramarine at low vs high reasoning on the same 200 tasks
  8. crystal balls    two models asked "does it exist?": opposite biases
  9. cauldron         what kind of names the ghosts are
 10. cemetery gate    the guard: replay over every generation, and hook latency
 11. cover            the post's cover image, 2000 x 840
Every number is read from results/summary.json; no ghost name is drawn anywhere
(the cover's terminal shows a redaction bar). validate() fails loudly on overlaps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

from phantom import config

_spec = importlib.util.spec_from_file_location("hf", Path(__file__).with_name("07b_halloween_figures.py"))
hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf)
fp, outline, DECOR = hf.fp, hf.outline, hf.DECOR


def mini_ghost(x: float, y: float, s: float) -> np.ndarray:
    """Outline of a small sheet ghost: width s, from y - 0.45 s to y + 1.05 s."""
    t = np.linspace(np.pi, 0, 24)
    top = np.c_[x + 0.5 * s * np.cos(t), y + 0.55 * s + 0.5 * s * np.sin(t)]
    hx = np.linspace(x + 0.5 * s, x - 0.5 * s, 13)
    hem = np.c_[hx, y - 0.45 * s + 0.12 * s * np.abs(np.sin((hx - x) / s * 3 * np.pi))]
    return np.vstack([top, hem])


def draw_ghost(ax, x, y, s, color="#eef0ff", alpha=1.0, z=3.0, eyes="#14102a", dashed=False):
    ax.add_patch(Polygon(mini_ghost(x, y, s), closed=True, facecolor=color if not dashed else "none",
                         edgecolor="#b9b2e6" if dashed else "none", alpha=alpha, lw=1.0,
                         linestyle=(0, (2, 2)) if dashed else "-", zorder=z))
    for dx in (-0.17, 0.17):
        ax.add_patch(Ellipse((x + dx * s, y + 0.62 * s), 0.13 * s, 0.2 * s, color=eyes, alpha=alpha, zorder=z + 0.1,
                             lw=0))


# ---------------------------------------------------------------- 6. haunted mirror

def mirror(summary: dict, problems: dict) -> None:
    hf.set_body("mysteryquest")
    fig, bg = hf.canvas(["#24163a", "#120b22", "#05030a"])
    hf.stars(bg, n=70, ymin=60, color="#e6dcff", seed=17)
    cx, cy, rx, ry = 46, 41, 29, 35
    hf.radial_glow(bg, cx, cy, rx * 1.7, "#8f7cff", alpha=0.25, z=1.15, ry=ry * 1.5)
    glass = Ellipse((cx, cy), 2 * rx, 2 * ry, facecolor="none", edgecolor="none", zorder=1.3)
    bg.add_patch(glass)
    hf.fill_gradient(bg, glass, [cx - rx, cx + rx, cy - ry, cy + ry], "#141a33", "#3a4777", 1.3, noise=0.02, seed=3)
    for scale, lw, col in ((1.0, 13, "#6e4c1c"), (1.0, 7, "#c99a45"), (0.955, 1.6, "#e6c27a")):  # gilt frame
        bg.add_patch(Ellipse((cx, cy), 2 * rx * scale, 2 * ry * scale, fill=False, edgecolor=col, lw=lw, zorder=1.6))
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):  # frame studs
        bg.add_patch(Circle((cx + rx * 1.035 * np.cos(a), cy + ry * 1.03 * np.sin(a)), 0.9, color="#e6c27a",
                            zorder=1.65, lw=0))

    rep = summary["reproducibility"]
    n_total, n_back = rep["ghosts"]["names_run1"], rep["ghosts"]["names_in_both"]
    pts = []
    for step in np.linspace(4.6, 3.0, 40):  # the widest grid that still fits every run-1 ghost in the glass
        xs = np.arange(cx - rx, cx + rx + 0.01, step)
        ys = np.arange(cy + ry, cy - ry - 0.01, -step * 1.12)
        pts = [(x, y) for y in ys for x in xs if ((x - cx) / (rx * 0.86)) ** 2 + ((y - cy) / (ry * 0.86)) ** 2 <= 1]
        if len(pts) >= n_total:
            break
    pts = pts[:n_total]
    size = step * 0.62
    ax = bg
    for i, (x, y) in enumerate(pts):
        lost = i == len(pts) - 1 and n_back < n_total
        if lost:
            draw_ghost(ax, x, y - 0.3 * size, size, alpha=0.5, z=2.0, dashed=True, eyes="#b9b2e6")
        else:
            draw_ghost(ax, x, y - 0.3 * size, size, color="#f1efff", z=2.0)
    hf.vignette(bg, 0.5)
    hf.grain(bg, alpha=0.035)
    fig._keepout.append(("mirror", (cx - rx - 1.5, cy - ry - 1.5, cx + rx + 1.5, cy + ry + 1.5)))

    g = rep["generations"]
    pm = g["per_model"]
    hf.panel(fig, 0.79, [
        (f"{n_back} of {n_total}", "creepster", 60, "#d9ceff", 0.14),
        ("ghost names came back when I\nreran the whole study from scratch", "body", 21, "#efe9ff", 0.15),
        (f"{g['identical_outputs']:,} of {g['common_keys']:,} outputs were\nidentical, word for word "
         f"({g['identical_share']:.1%})", "body", 19, "#d2c9f0", 0.13),
        ("\n".join(f"{m.split(':')[0]}: {v['identical']} of {v['total']} identical" for m, v in pm.items()),
         "body_r", 17, "#bdb4de", 0.17),
        ("Each ghost in the mirror is one\nname from run 1. The faint, dashed\none did not come back in run 2.",
         "body_r", 16, "#a99fcc", 0.0)], x=0.53)
    hf.title(fig, "Mirror, mirror: same ghosts", "Run 1 vs. run 2 · same 200 prompts, 3 models, seeds "
             "1001/1002, temperature 0.7 · names withheld", "eater", "#ece6ff", "#cfc6ea", 46, 0.835,
             top=0.955, sub_size=17)
    hf.footer(fig, "#9d93c0")
    hf.save(fig, "halloween_reproducibility.png", problems)


# ---------------------------------------------------------------- 7. potion shelf

def bottle(x: float, top: float) -> np.ndarray:
    return np.array([(x - 0.3, 0), (x + 0.3, 0), (x + 0.3, top * 0.8), (x + 0.11, top * 0.9), (x + 0.11, top),
                     (x - 0.11, top), (x - 0.11, top * 0.9), (x - 0.3, top * 0.8)])


def potions(summary: dict, problems: dict) -> None:
    hf.set_body("hennypenny")
    fig, bg = hf.canvas(["#0f2a26", "#08161a", "#030809"])
    hf.stars(bg, n=50, ymin=70, color="#d8fff0", seed=21)
    bg.add_patch(Rectangle((0, 12.5), 160, 2.2, color="#3b2414", zorder=1.5))  # the shelf
    bg.add_patch(Rectangle((0, 12.2), 160, 0.5, color="#1c0f07", zorder=1.51))
    for x, h, c in ((150, 7, "#7a3cff"), (154.5, 5, "#ff7a2a"), (8, 6, "#3fe0a0")):  # side vials (decor)
        bg.add_patch(Polygon(bottle(x, h) / [1, 1] + [0, 14.7], closed=True, facecolor=c, alpha=0.35,
                             edgecolor="#cfe8e0", lw=1.2, zorder=1.55))
    hf.fog(bg, 0, 160, 0, 14, "#5fa58f", 0.18, 4, 1.6)
    hf.vignette(bg)
    hf.grain(bg)

    ab = summary["hosted"]["ablation"]
    models = summary["hosted"]["models"]
    rows = [("ultramarine-1-flash", "low reasoning", "#a46bff"), ("ultramarine-1-flash@high", "high reasoning",
                                                                   "#3fe0a0")]
    ymax = 0.05
    rect = [hf.PLOT_X, 0.2, 0.45, 0.56]
    ax = hf.data_axes(fig, rect, "#d8efe6", "#1d3a34", (-0.6, 1.6), (0, ymax))
    ax.yaxis.set_major_formatter(hf.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    labels = []
    for i, (m, name, color) in enumerate(rows):
        r = ab["rates"][m]
        v = r["ghost_rate"]
        top = ymax * 0.93
        glass = Polygon(bottle(i, top), closed=True, facecolor="#cfe8e0", alpha=0.10, edgecolor="none", zorder=3)
        ax.add_patch(glass)
        liquid = Polygon(bottle(i, top), closed=True, facecolor="none", edgecolor="none", zorder=3.5)
        ax.add_patch(liquid)
        im = hf.fill_gradient(ax, liquid, [i - 0.3, i + 0.3, 0, v], color, "#ffffff", 3.5, horizontal=True,
                              center=0.4, noise=0.02, seed=i)
        im.set_clip_path(liquid)
        ax.add_patch(Rectangle((i - 0.3, v - 0.0002), 0.6, 0.0004, color="#ffffff", zorder=3.7))  # meniscus
        ax.add_patch(Polygon(bottle(i, top), closed=True, facecolor="none", edgecolor="#e6fff6", lw=2.4, zorder=4))
        ax.add_patch(Rectangle((i - 0.09, top), 0.18, 0.003, color="#8a5a32", zorder=4))  # cork
        hf.radial_glow(ax, i, v / 2 + 0.002, 0.5, color, alpha=0.3, z=2.5, ry=0.012)
        ax.text(i + 0.36, v, f"{v:.1%}", ha="left", va="center", color="#ffffff", zorder=10,
                path_effects=outline(5), **fp("body", 30))
        labels.append(f"{name}\n{r['ghosts']} of {r['mentions']} installs")
    ax.set_xticks(range(len(rows)), labels)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=10)
    hf.ticks(ax, "#e6fff6")
    hf.axis_label(ax, "y", "install suggestions that don't exist on PyPI", "#d8efe6")
    lo, hi = models["ultramarine-1-flash"], models["ultramarine-1-flash@high"]
    sp = summary["hosted"]["spend"]["parts"]
    local = summary["rates"]["per_model"]
    lmin, lmax = min(v["ghost_rate"] for v in local.values()), max(v["ghost_rate"] for v in local.values())
    hf.panel(fig, 0.77, [
        (f"{ab['only_low']} tasks", "nosifer", 34, "#c9a2ff", 0.11),
        (f"ghosted only at low effort;\n{ab['only_high']} only at high, {ab['both']} at both",
         "body", 19, "#e6fff6", 0.145),
        (f"cost per 1,000 generations\nlow \\${lo['cost_per_1k_generations_usd']:.2f} · "
         f"high \\${hi['cost_per_1k_generations_usd']:.2f}", "body", 18, "#bfe9da", 0.14),
        (f"reasoning tokens\nlow {sp['A_low']['reasoning_tokens']:,} · high {sp['B_high']['reasoning_tokens']:,}",
         "body", 18, "#bfe9da", 0.14),
        (f"For scale, the three local models:\n{lmin:.1%} to {lmax:.1%}. Hosted, not open:\n"
         "a comparison, not pooled.", "body_r", 15, "#9fcabb", 0.0)], x=0.64)
    hf.title(fig, "More thinking, fewer ghosts", f"Ultramarine-1-Flash on the same {ab['n_prompts']} tasks "
             f"(sample {ab['sample']}) at two reasoning efforts", "nosifer", "#d9fff0", "#bfe9da", 40, 0.835,
             top=0.95)
    hf.footer(fig, "#8fb8a8")
    hf.save(fig, "halloween_reasoning_potions.png", problems)


# ---------------------------------------------------------------- 8. crystal balls

def crystal_ball(bg, x, y, r, glow):
    hf.radial_glow(bg, x, y, r * 1.9, glow, alpha=0.35, z=1.2)
    bg.add_patch(Polygon([(x - r * 0.75, y - r * 1.25), (x + r * 0.75, y - r * 1.25), (x + r * 0.5, y - r * 0.85),
                          (x - r * 0.5, y - r * 0.85)], closed=True, color="#2a1a10", ec="#a07a3a", lw=2, zorder=1.4))
    ball = Circle((x, y), r, facecolor="none", edgecolor="none", zorder=1.5)
    bg.add_patch(ball)
    hf.fill_gradient(bg, ball, [x - r, x + r, y - r, y + r], "#120d26", glow, 1.5, center=0.5, noise=0.015, seed=8)
    bg.add_patch(Circle((x, y), r, fill=False, edgecolor="#efe6ff", lw=2.5, alpha=0.8, zorder=1.6))
    bg.add_patch(Ellipse((x - r * 0.4, y + r * 0.5), r * 0.45, r * 0.25, angle=30, color="#ffffff", alpha=0.25,
                         zorder=1.6, lw=0))


def crystal(summary: dict, problems: dict) -> None:
    hf.set_body("mysteryquest")
    fig, bg = hf.canvas(["#1d1032", "#0e0820", "#04020a"])
    hf.stars(bg, n=90, ymin=40, color="#efe4ff", seed=5)
    sc, hs = summary["selfcheck"], summary["hosted"]["selfcheck"]
    ev = summary["eval"]["A"]
    balls = [(40, "#ff8a3c", "gemma4:e4b · local, open", "NO", f"to {sc['real_names_said_missing']} of "
              f"{sc['real_names']}\nreal packages"),
             (120, "#a77bff", "Ultramarine-1-Flash · hosted", "YES", f"to {hs['ghost_names_said_exists']} of "
              f"{hs['ghost_names']}\nghosts")]
    for x, glow, name, word, sub in balls:
        crystal_ball(bg, x, 59, 13, glow)
        fig.text(x / 160, 64.5 / 100, word, ha="center", va="center", color="#ffffff", path_effects=outline(6),
                 **fp("creepster", 46))
        fig.text(x / 160, 56 / 100, sub, ha="center", va="top", color="#f4ecff", linespacing=1.15,
                 path_effects=outline(4), **fp("body", 18))
        fig.text(x / 160, 40.5 / 100, name, ha="center", va="top", color=glow, path_effects=outline(4),
                 **fp("body", 19))
    fig.text(0.5, 67 / 100, "AUROC on split A", ha="center", va="top", color="#d8ccf4",
             linespacing=1.2, path_effects=outline(4), **fp("body_r", 16))
    fig.text(0.5, 62 / 100, f"gemma4, by P(no)  {ev['llm_selfcheck_prob']['auroc']:.2f}\n"
             f"Ultramarine, by P(no)  {ev['hosted_selfcheck_prob']['auroc']:.2f}\n"
             f"TabPFN, no model call  {ev['tabpfn']['auroc']:.2f}", ha="center", va="top", color="#f4ecff",
             linespacing=1.4, path_effects=outline(4), **fp("body", 17))
    hf.vignette(bg)
    hf.grain(bg)

    rect = [0.3, 0.08, 0.6, 0.23]
    ax = hf.data_axes(fig, rect, "#ddd2f4", "#2a2042", (0, 0.68), (-0.6, 1.6))
    ax.xaxis.set_major_formatter(hf.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="y", visible=False)
    rows = [("said a real\npackage is missing", "real_names_said_missing", "real_names"),
            ("said a ghost\nexists", "ghost_names_said_exists", "ghost_names")]
    for j, (_, num, den) in enumerate(rows):
        yy = 1 - j
        for (src, color, label, off) in ((sc, "#ff8a3c", "gemma4", 0.2), (hs, "#a77bff", "Ultramarine", -0.2)):
            v = src[num] / src[den]
            ax.barh(yy + off, v, height=0.36, color=color, edgecolor="#120a20", lw=1.2, zorder=4)
            ax.text(v + 0.008, yy + off, f"{label}: {src[num]} of {src[den]} ({v:.0%})", va="center", ha="left",
                    color="#f4ecff", zorder=8, path_effects=outline(4), **fp("body", 15))
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, pad=10)
    hf.ticks(ax, "#ece4ff")
    hf.title(fig, "Ask a model if it exists", 'Each model was asked "Does a package named X exist on PyPI? '
             f'Answer only yes or no." for {sc["n_names"]} names: {sc["ghost_names"]} ghosts, '
             f"{sc['real_names']} real", "creepster", "#f1e8ff", "#d2c6ee", 48, 0.86, top=0.96, sub_size=16)
    hf.footer(fig, "#9c90c0")
    hf.save(fig, "halloween_selfcheck.png", problems)


# ---------------------------------------------------------------- 9. cauldron

TYPE_LABELS = {"task_words": "Words from the task, glued", "invented": "Invented outright",
               "affix_of_real": "A real name plus an affix", "near_real_name": "One or two edits from a real name",
               "import_or_brand": "An import name or a brand"}


def cauldron_art(bg, x, y, s):
    for dx, h in ((-0.6, 1.2), (0, 1.6), (0.6, 1.1)):  # fire
        bg.add_patch(Polygon(hf.flame_shape(x + dx * s, y - 1.25 * s, 0.7 * s, h * s), closed=True, color="#ff7a1a",
                             alpha=0.9, zorder=1.45, lw=0))
        bg.add_patch(Polygon(hf.flame_shape(x + dx * s, y - 1.25 * s, 0.35 * s, h * 0.6 * s), closed=True,
                             color="#ffe07a", zorder=1.46, lw=0))
    hf.radial_glow(bg, x, y + 0.6 * s, 2.6 * s, "#7dff6a", alpha=0.35, z=1.42)
    bg.add_patch(Ellipse((x, y), 3.2 * s, 2.5 * s, color="#0b0b0e", zorder=1.5, lw=0))
    bg.add_patch(Ellipse((x, y + 0.95 * s), 3.0 * s, 0.7 * s, color="#58e05a", zorder=1.55, lw=0))
    bg.add_patch(Ellipse((x, y + 0.95 * s), 3.2 * s, 0.8 * s, fill=False, edgecolor="#2a2a30", lw=5, zorder=1.56))
    rng = np.random.default_rng(4)
    for _ in range(22):  # bubbles
        bx, by, br = x + rng.uniform(-1.2, 1.2) * s, y + rng.uniform(1.1, 3.6) * s, rng.uniform(0.08, 0.3) * s
        bg.add_patch(Circle((bx, by), br, fill=False, edgecolor="#a6ff8a", lw=1.2, alpha=0.7, zorder=1.57))


def cauldron(summary: dict, problems: dict) -> None:
    hf.set_body("griffy")
    fig, bg = hf.canvas(["#13230f", "#0a1408", "#030603"])
    hf.stars(bg, n=60, ymin=70, color="#eaffd8", seed=9)
    cauldron_art(bg, 140, 12, 5.2)
    fig._keepout.append(("cauldron", (122, 4, 158, 32)))
    hf.fog(bg, 0, 160, 0, 12, "#7fa86a", 0.2, 5, 1.6)
    hf.vignette(bg)
    hf.grain(bg)

    gt = summary["ghost_types"]
    order = sorted(gt["by_type"], key=lambda t: -gt["by_type"][t]["n_names"])
    rect = [0.33, 0.14, 0.4, 0.58]
    ax = hf.data_axes(fig, rect, "#e2f4d6", "#22361a", (0, 0.75), (-0.6, len(order) - 0.4))
    ax.xaxis.set_major_formatter(hf.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="y", visible=False)
    for j, t in enumerate(order):
        d = gt["by_type"][t]
        yy = len(order) - 1 - j
        bar = Rectangle((0, yy - 0.3), d["share_of_names"], 0.6, facecolor="none", edgecolor="none", zorder=4)
        ax.add_patch(bar)
        hf.fill_gradient(ax, bar, [0, d["share_of_names"], yy - 0.3, yy + 0.3], "#2f8a2a", "#b6ff8a", 4,
                         horizontal=True, noise=0.02, seed=j)
        ax.add_patch(Rectangle((0, yy - 0.3), d["share_of_names"], 0.6, fill=False, edgecolor="#0d1a08", lw=1.6,
                               zorder=4.1))
        ax.text(d["share_of_names"] + 0.012, yy, f"{d['n_names']} names · {d['share_of_names']:.0%}", va="center",
                ha="left", color="#f2ffe8", zorder=8, path_effects=outline(4), **fp("body", 18))
    ax.set_yticks(range(len(order))[::-1], [TYPE_LABELS[t] for t in order])
    ax.tick_params(axis="y", length=0, pad=10)
    hf.ticks(ax, "#eaffdc", size=16)
    hf.axis_label(ax, "x", "share of the ghost names (first rule that fits)", "#e2f4d6")
    tw = gt["by_type"]["task_words"]
    hf.panel(fig, 0.73, [
        (f"{tw['repeated_2plus']} of {summary['repeat']['repeated_2plus']}", "creepster", 44, "#b6ff8a", 0.105),
        ("repeated ghosts are\nwords from the task", "body", 20, "#eaffdc", 0.12),
        ("Ask for a script that reads\nformat X and the model\nreaches for a package\nnamed after X.",
         "body_r", 16, "#c8e6b8", 0.0)], x=0.79)
    hf.title(fig, "What's in the cauldron?", f"How the {gt['n_ghost_names']} ghost names were built · names "
             "withheld · rules in results/summary.json", "eater", "#e6ffd6", "#cde8bd", 44, 0.825, top=0.955)
    hf.footer(fig, "#9cba8c")
    hf.save(fig, "halloween_ghost_types.png", problems)


# ---------------------------------------------------------------- 10. cemetery gate

def fence(bg, y0, y1, color="#2b3148"):
    bg.add_patch(Rectangle((0, y0), 160, 0.9, color=color, zorder=1.6))
    bg.add_patch(Rectangle((0, y1 - 2.2), 160, 0.7, color=color, zorder=1.6))
    for x in np.arange(1.5, 160, 3.2):
        bg.add_patch(Rectangle((x - 0.3, y0), 0.6, y1 - y0, color=color, zorder=1.6))
        bg.add_patch(Polygon([(x - 0.8, y1), (x + 0.8, y1), (x, y1 + 2.0)], color=color, zorder=1.6))


def gate(summary: dict, problems: dict) -> None:
    hf.set_body("specialelite")
    fig, bg = hf.canvas(["#1a1f2e", "#0c0f18", "#040509"])
    hf.stars(bg, n=110, ymin=55, color="#e8ecff", seed=13)
    hf.moon(fig, bg, 148, 88, 6, glow_size=2.6)
    for x, y, s in ((128, 90, 1.8), (136, 84, 1.2)):
        hf.bat(bg, x, y, s)
    hf.fog(bg, 0, 160, 0, 14, "#8a96b8", 0.25, 2, 1.55)
    fence(bg, 0, 8)
    hf.vignette(bg)
    hf.grain(bg)

    rp = summary["replay"]["overall"]
    on, off = rp["online"], rp["offline"]
    n_ghost, n_real = rp["ghost_mentions"], rp["real_mentions"]
    ax = hf.data_axes(fig, [0.075, 0.2, 0.4, 0.5], "#dfe4f2", "#232a3c", (-0.6, 1.6), (0, 1.12))
    ax.yaxis.set_major_formatter(hf.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bars = [(0, on["ghosts_blocked"] / n_ghost, f"{on['ghosts_blocked']} of {n_ghost}", "#ff9a3c", -0.19),
            (0, off["ghosts_caught"] / n_ghost, f"{off['ghosts_caught']} of {n_ghost}", "#8f9fd8", 0.19),
            (1, on["real_blocked"] / n_real, f"{on['real_blocked']} of\n{n_real:,}", "#ff9a3c", -0.19),
            (1, off["real_flagged"] / n_real, f"{off['real_flagged']} of\n{n_real:,}", "#8f9fd8", 0.19)]
    for x, v, lab, c, off_x in bars:
        ax.bar(x + off_x, v, width=0.34, color=c, edgecolor="#0a0c14", lw=1.4, zorder=4)
        ax.text(x + off_x, v + 0.025, f"{v:.1%}\n{lab}", ha="center", va="bottom", color="#f2f4ff", zorder=8,
                linespacing=1.15, path_effects=outline(4), **fp("body", 14))
    ax.set_xticks([0, 1], ["ghost installs stopped", "real installs stopped"])
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=10)
    hf.ticks(ax, "#eef1ff")
    fig.text(0.075, 0.755, "Replay over all 1,199 generations", color="#f2f4ff", path_effects=outline(4),
             **fp("body", 21))
    for k, (c, lab) in enumerate((("#ff9a3c", "online: asks PyPI first"), ("#8f9fd8", "offline: TabPFN score only"))):
        fig.text(0.095 + 0.2 * k, 0.715, lab, color=c, path_effects=outline(4), **fp("body", 15))
        fig.patches.append(Rectangle((0.08 + 0.2 * k, 0.717), 0.011, 0.016, color=c, transform=fig.transFigure,
                                     figure=fig))

    lat = summary["hook_latency"]["cases"]
    cases = [("no_install", "no install in the command"), ("real_install", "real package install"),
             ("ghost_install_default", "ghost install (PyPI first)"),
             ("ghost_install_with_p_ghost", "ghost install, with TabPFN score")]
    ax2 = hf.data_axes(fig, [0.66, 0.2, 0.29, 0.5], "#dfe4f2", "#232a3c", (0, 16), (-0.6, 3.6))
    ax2.grid(axis="y", visible=False)
    for j, (key, _) in enumerate(cases):
        v = lat[key]["median_s"]
        yy = 3 - j
        ax2.barh(yy, v, height=0.55, color="#c9cfe6" if j < 3 else "#6f7aa6", edgecolor="#0a0c14", lw=1.4,
                 zorder=4)
        ax2.text(v + 0.3, yy, f"{v:.1f} s", va="center", ha="left", color="#f2f4ff", zorder=8,
                 path_effects=outline(4), **fp("body", 17))
    ax2.set_yticks([3, 2, 1, 0], [c[1] for c in cases])
    ax2.tick_params(axis="y", length=0, pad=8)
    hf.ticks(ax2, "#eef1ff", size=14)
    hf.axis_label(ax2, "x", f"seconds per command (median of {summary['hook_latency']['runs_per_case']})",
                  "#dfe4f2", size=14)
    fig.text(0.95, 0.755, "Claude Code hook latency", ha="right", color="#f2f4ff", path_effects=outline(4),
             **fp("body", 21))
    fig.text(0.95, 0.715, "Windows 11 laptop, CPU only", ha="right", color="#b8bfd8", path_effects=outline(4),
             **fp("body_r", 15))
    hf.title(fig, "Nothing gets past the gate", "The guard replayed over every install in the study, and what "
             "it costs per command", "metalmania", "#eef1ff", "#cdd3e8", 54, 0.85, top=0.96)
    hf.footer(fig, "#8d95b0")
    hf.save(fig, "halloween_guard.png", problems)


# ---------------------------------------------------------------- 11. cover

def cover(summary: dict, problems: dict) -> None:
    hf.set_body("specialelite")
    fig, bg = hf.canvas(["#2a1340", "#120a22", "#05030a"], hu=42, wu=100, width_in=2000 / hf.DPI)
    hf.stars(bg, n=110, ymin=14, color="#f3eaff", seed=44)
    hf.moon(fig, bg, 91, 34, 4.2, glow_size=3.0)
    for x, y, s, f in ((40, 38, 1.4, 0.1), (46, 35, 0.9, -0.1), (79, 39, 1.0, 0.05)):
        hf.bat(bg, x, y, s, flap=f)
    hf.fog(bg, 0, 100, 0, 9, "#a28cd8", 0.3, 7, 1.6)

    tx, ty, tw, th = 55, 4.5, 37, 21  # the terminal
    hf.radial_glow(bg, tx + tw / 2, ty + th + 4, 18, "#b9a8ff", alpha=0.3, z=1.65)
    draw_ghost(bg, tx + tw * 0.62, ty + th + 1.6, 9.5, color="#f4f1ff", z=1.68)
    bg.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=1.2", facecolor="#0c0b12",
                                edgecolor="#5c5478", lw=2.5, zorder=1.7))
    bg.add_patch(FancyBboxPatch((tx, ty + th - 2.4), tw, 2.4, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor="#221d33", edgecolor="none", zorder=1.71))
    for k, c in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        bg.add_patch(Circle((tx + 1.6 + 1.4 * k, ty + th - 1.2), 0.42, color=c, zorder=1.72))
    hf.vignette(bg, 0.5)
    hf.grain(bg, alpha=0.03)

    def term(x, y, s, color, size):
        fig.text(x / 100, y / 42, s, color=color, va="center", ha="left", **fp("body", size))

    term(tx + 1.6, ty + 15.2, "$ pip install", "#e8e4ff", 27)
    bg.add_patch(Rectangle((tx + 17.2, ty + 14.1), 14.5, 2.2, color="#e8e4ff", zorder=1.75))  # redacted name
    term(tx + 1.6, ty + 11.0, "BLOCKED: not on PyPI.", "#ff7a59", 27)
    term(tx + 1.6, ty + 7.4, "Nobody ever wrote it.", "#b9b2d8", 23)
    term(tx + 1.6, ty + 3.8, "Nothing was installed.", "#7ee2a8", 23)

    r = summary["rates"]["overall"]
    rep = summary["reproducibility"]["ghosts"]
    fig.text(0.045, 0.83, "My local AI invented", color="#f2e9ff", va="top", path_effects=outline(6),
             **fp("creepster", 54))
    fig.text(0.045, 0.64, f"{rep['names_run1']} Python packages", color="#ffb347", va="top",
             path_effects=outline(7), **fp("creepster", 70))
    fig.text(0.045, 0.39, f"I reran everything. {rep['names_in_both']} came back.", color="#e2d8ff", va="top",
             path_effects=outline(5), **fp("body", 29))
    fig.text(0.045, 0.27, f"{r['ghost_rate']:.0%} of install suggestions pointed at ghosts.", color="#c9bfe8",
             va="top", path_effects=outline(5), **fp("body", 24))
    hf.save(fig, "cover.png", problems)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()
    hf.OUT.mkdir(exist_ok=True)
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    problems: dict[str, list[str]] = {}
    mirror(summary, problems)
    if summary.get("hosted", {}).get("complete"):
        potions(summary, problems)
        crystal(summary, problems)
    cauldron(summary, problems)
    gate(summary, problems)
    cover(summary, problems)
    bad = {k: v for k, v in problems.items() if v}
    if bad:
        raise SystemExit(f"layout validation failed: {sum(map(len, bad.values()))} issue(s) in {list(bad)}")
    print("all figures passed layout validation")


if __name__ == "__main__":
    main()
