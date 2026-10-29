"""Halloween "costume" versions of the result figures for the blog, saved to output/.

Each figure wears a different costume; decoration never carries data. Every bar, whisker,
line and point is drawn from the same committed results as experiments/07_figures.py:
  1. graveyard        ghost rate by model: textured tombstones under a full moon
  2. candle ritual    ghost rate by niche: candles that burn higher on obscure tasks
  3. spider web       calibration: each TabPFN bin is a spider hanging on its thread
  4. ouija board      precision-recall: the yes/no baselines sit where the planchette stops
  5. pumpkin patch    repeat histogram: a parade of ghosts over jack-o'-lanterns
After drawing, validate() checks that no two labels overlap, that no text leaves the
canvas, and that no label sits on a "keep-out" decoration (moon, house, trees); it fails loudly.
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
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from phantom import config  # noqa: E402

OUT = config.ROOT / "output"
W, H, DPI = 16, 10, 120  # 1920 x 1200 px; background axes use 0..160 x 0..100 units
MARGIN = 0.055           # left edge of titles; plots start at PLOT_X so tick labels line up under it
PLOT_X = 0.105
PANEL_X = 0.76           # right-hand text column
MODELS = ["gemma4:e4b", "phi4-mini:3.8b", "qwen3:4b-instruct"]
MODEL_COLORS = {"gemma4:e4b": "#e0703f", "phi4-mini:3.8b": "#a79cf0", "qwen3:4b-instruct": "#3fbf8f"}
FONTS = {k: f"C:/Windows/Fonts/{v}" for k, v in {
    "chiller": "CHILLER.TTF", "onyx": "ONYX.TTF", "viner": "VINERITC.TTF", "georgia": "georgiab.ttf",
    "georgia_r": "georgia.ttf", "jokerman": "JOKERMAN.TTF", "body": "segoeuib.ttf", "body_r": "segoeui.ttf",
    "body_sb": "seguisb.ttf"}.items()}
DECOR = "decor"  # gid for decorative text the validator ignores (board letters, runes)


# ---------------------------------------------------------------- primitives

def fp(key: str, size: float) -> dict:
    """Font kwargs for a Windows font file (falls back to Segoe UI Bold, then the default)."""
    for k in (key, "body"):
        try:
            return {"fontproperties": font_manager.FontProperties(fname=FONTS[k], size=size)}
        except (OSError, KeyError, ValueError):
            continue
    return {"fontsize": size}


def outline(width: float = 4, color: str = "#07040a") -> list:
    return [pe.withStroke(linewidth=width, foreground=color)]


def rgba_image(rgb, alpha) -> np.ndarray:
    img = np.zeros((*alpha.shape, 4))
    img[..., :3] = rgb
    img[..., 3] = np.clip(alpha, 0, 1)
    return img


def radial_glow(ax, x, y, rx, color, alpha=0.6, z=1.0, ry=None, power=2.0):
    """Soft radial light, brightest in the centre."""
    n = 220
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    a = np.clip(1 - np.sqrt(xx ** 2 + yy ** 2), 0, 1) ** power * alpha
    ry = rx if ry is None else ry
    ax.imshow(rgba_image(to_rgb(color), a), extent=[x - rx, x + rx, y - ry, y + ry], zorder=z,
              interpolation="bilinear", aspect="auto")


def fill_gradient(ax, patch, extent, c0, c1, z, horizontal=False, noise=0.0, seed=0, center=None):
    """Fill `patch` (already added to ax) with a linear gradient c0->c1, optional grain."""
    n = 300
    g = np.linspace(0, 1, n)
    t = np.tile(g, (n, 1)) if horizontal else np.tile(g[:, None], (1, n))
    if center is not None:  # bright band at `center` (0..1 across), darker towards both edges
        t = np.clip(1 - np.abs(np.tile(g, (n, 1)) - center) * 1.8, 0, 1)
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    img = a + (b - a) * t[..., None]
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise, (n, n))[..., None]
    im = ax.imshow(np.clip(img, 0, 1), extent=extent, origin="lower", aspect="auto", zorder=z,
                   interpolation="bilinear")
    im.set_clip_path(patch)
    return im


def smooth_noise(nx, ny, seed, octaves=4):
    rng = np.random.default_rng(seed)
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)[:, None]
    f = np.zeros((ny, nx))
    for o in range(octaves):
        k = 2 ** o
        f += np.sin(2 * np.pi * (k * 1.3 * x + rng.uniform())) * np.cos(2 * np.pi * (k * 0.7 * y + rng.uniform())) / k
    return (f - f.min()) / (f.max() - f.min())


def fog(ax, x0, x1, y0, y1, color, alpha, seed, z):
    """A soft, uneven fog band (densest in the middle of the band)."""
    f = smooth_noise(500, 90, seed)
    band = np.sin(np.linspace(0, np.pi, 90))[:, None] ** 1.5
    ax.imshow(rgba_image(to_rgb(color), f * band * alpha), extent=[x0, x1, y0, y1], origin="lower",
              zorder=z, interpolation="bilinear", aspect="auto")


def canvas(stops: list[str]):
    """Figure with a full-bleed background axes (0..160 x 0..100) and a vertical gradient."""
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.set_xlim(0, 160)
    bg.set_ylim(0, 100)
    bg.set_autoscale_on(False)
    bg.axis("off")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bg", stops[::-1])
    bg.imshow(np.linspace(0, 1, 512)[:, None], extent=[0, 160, 0, 100], origin="lower", cmap=cmap,
              aspect="auto", zorder=0)
    fig._keepout = []  # background-unit boxes that labels must not cover
    return fig, bg


def vignette(bg, strength=0.55):
    yy, xx = np.mgrid[-1:1:300j, -1:1:480j]
    a = np.clip(np.sqrt((xx * 0.9) ** 2 + (yy * 1.05) ** 2) - 0.55, 0, 1) ** 1.6 * strength
    bg.imshow(rgba_image((0, 0, 0), a), extent=[0, 160, 0, 100], zorder=1.8, interpolation="bilinear",
              aspect="auto")


def grain(bg, seed=1, alpha=0.05):
    n = np.random.default_rng(seed).random((1200, 1920))  # one value per output pixel: fine, not blocky
    bg.imshow(rgba_image((1, 1, 1), (n - 0.5) * 2 * alpha), extent=[0, 160, 0, 100], zorder=1.9,
              interpolation="antialiased", aspect="auto")


def stars(bg, n=160, ymin=50, color="#fff6e0", seed=31):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 160, n), rng.uniform(ymin, 100, n)
    s = rng.power(3, n) * 9 + 0.4
    bg.scatter(x, y, s=s, color=color, alpha=0.75, linewidths=0, zorder=1.1)
    for i in np.argsort(s)[-6:]:  # a few twinkles
        bg.plot([x[i] - 0.7, x[i] + 0.7], [y[i], y[i]], color=color, lw=0.8, alpha=0.6, zorder=1.1)
        bg.plot([x[i], x[i]], [y[i] - 0.7, y[i] + 0.7], color=color, lw=0.8, alpha=0.6, zorder=1.1)


def moon(fig, bg, x, y, r, c0="#fff6dc", c1="#d8cba4", glow="#f6eccc", glow_size=3.2):
    radial_glow(bg, x, y, r * glow_size, glow, alpha=0.35, z=1.2, power=2.2)
    disc = Circle((x, y), r, facecolor="none", edgecolor="none", zorder=1.3)
    bg.add_patch(disc)
    fill_gradient(bg, disc, [x - r, x + r, y - r, y + r], c1, c0, 1.3, noise=0.015, seed=4)
    rng = np.random.default_rng(3)
    for _ in range(8):
        cx, cy = x + rng.uniform(-0.6, 0.6) * r, y + rng.uniform(-0.6, 0.6) * r
        if (cx - x) ** 2 + (cy - y) ** 2 < (0.8 * r) ** 2:
            bg.add_patch(Circle((cx, cy), r * rng.uniform(0.06, 0.16), color=c1, alpha=0.55, zorder=1.35, lw=0))
    fig._keepout.append(("moon", (x - r, y - r, x + r, y + r)))


BAT = np.array([(0.0, 0.18), (0.18, 0.32), (0.45, 0.42), (0.75, 0.58), (1.0, 0.62), (0.9, 0.38), (0.78, 0.22),
                (0.68, 0.30), (0.58, 0.08), (0.46, 0.16), (0.36, -0.04), (0.24, 0.06), (0.12, -0.08), (0.04, -0.02)])


def bat(ax, x, y, s, color="#08050a", z=1.4, flap=0.0):
    wing = BAT.copy()
    wing[:, 1] = wing[:, 1] + flap * wing[:, 0] ** 2
    right = wing * s + (x, y)
    left = wing * (-s, s) + (x, y)
    ax.add_patch(Polygon(np.vstack([right, left[::-1]]), closed=True, color=color, zorder=z, lw=0))
    ax.add_patch(Ellipse((x, y + 0.1 * s), 0.22 * s, 0.42 * s, color=color, zorder=z, lw=0))
    for side in (-1, 1):  # ears
        ax.add_patch(Polygon([(x + side * 0.03 * s, y + 0.27 * s), (x + side * 0.1 * s, y + 0.42 * s),
                              (x + side * 0.11 * s, y + 0.25 * s)], color=color, zorder=z, lw=0))


def dead_tree(ax, x, y, length, angle, depth, width, color, rng, z=1.5):
    if depth == 0 or length < 0.25:
        return
    x2, y2 = x + length * np.cos(angle), y + length * np.sin(angle)
    ax.plot([x, x2], [y, y2], color=color, lw=width, solid_capstyle="round", zorder=z)
    for da in (rng.uniform(0.25, 0.6), -rng.uniform(0.2, 0.55)):
        dead_tree(ax, x2, y2, length * rng.uniform(0.62, 0.8), angle + da + rng.normal(0, 0.08), depth - 1,
                  width * 0.66, color, rng, z)


def title(fig, text, sub, key, color, sub_color, size, sub_y):
    fig.text(MARGIN, 0.94, text, color=color, va="top", path_effects=outline(7), **fp(key, size))
    fig.text(MARGIN, sub_y, sub, color=sub_color, va="top", path_effects=outline(4), **fp("body_r", 18))


def footer(fig, color):
    fig.text(0.985, 0.014, "Phantom Packages · data: results/summary.json", ha="right", va="bottom",
             color=color, alpha=0.9, path_effects=outline(3), **fp("body_r", 12))


def data_axes(fig, rect, ink, grid, xlim, ylim):
    ax = fig.add_axes(rect, zorder=3)
    ax.set_facecolor("none")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_autoscale_on(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink)
        ax.spines[s].set_alpha(0.55)
    ax.grid(True, color=grid, linewidth=1, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink, length=4)
    return ax


def ticks(ax, ink, size=15):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(font_manager.FontProperties(fname=FONTS["body"], size=size))
        lab.set_color(ink)
        lab.set_path_effects(outline(3))


def axis_label(ax, which, text, color):
    kw = {"color": color, **fp("body_r", 16)}
    (ax.set_xlabel if which == "x" else ax.set_ylabel)(text, **kw)


def ci_whisker(ax, x, lo, hi, color, w):
    ax.plot([x, x], [lo, hi], color=color, linewidth=2.4, zorder=9, solid_capstyle="round", path_effects=outline(5))
    for v in (lo, hi):
        ax.plot([x - w, x + w], [v, v], color=color, linewidth=2.4, zorder=9, path_effects=outline(5))


def panel(fig, top, blocks, x=PANEL_X):
    """Right-hand column of text blocks, top-aligned at `top` (figure fraction)."""
    y = top
    for text, key, size, color, gap in blocks:
        t = fig.text(x, y, text, color=color, va="top", linespacing=1.3, path_effects=outline(5), **fp(key, size))
        y -= gap
    return t


# ---------------------------------------------------------------- validation

def validate(fig, name: str) -> list[str]:
    """Text must stay on canvas, not overlap other text, and not sit on keep-out decorations."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    hidden = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update({id(tick.label1), id(tick.label2)})
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_gid() != DECOR
             and id(t) not in hidden]
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0)) for t in texts]
    issues = []
    for t, b in boxes:
        if b.x0 < 2 or b.y0 < 2 or b.x1 > fw - 2 or b.y1 > fh - 2:
            issues.append(f"off-canvas: {t.get_text()[:40]!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (t1, b1), (t2, b2) = boxes[i], boxes[j]
            ov = Bbox.intersection(b1, b2)
            if ov is not None and ov.width > 3 and ov.height > 3:
                issues.append(f"overlap: {t1.get_text()[:30]!r} x {t2.get_text()[:30]!r}")
    for label, (x0, y0, x1, y1) in getattr(fig, "_keepout", []):
        kb = Bbox.from_extents(x0 / 160 * fw, y0 / 100 * fh, x1 / 160 * fw, y1 / 100 * fh)
        for t, b in boxes:
            ov = Bbox.intersection(b, kb)
            if ov is not None and ov.width > 3 and ov.height > 3:
                issues.append(f"on {label}: {t.get_text()[:40]!r}")
    print(f"  validate {name}: {len(texts)} labels, {len(issues)} issue(s)")
    for i in issues:
        print("    -", i)
    return issues


def save(fig, name: str, problems: dict) -> None:
    problems[name] = validate(fig, name)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 1. graveyard

def tombstone_path(x0, x1, y0, y1, cap):
    cx, rx = (x0 + x1) / 2, (x1 - x0) / 2
    t = np.linspace(np.pi, 0, 48)
    top = np.c_[cx + rx * np.cos(t), (y1 - cap) + cap * np.sin(t)]
    return MPath(np.vstack([[x0, y0], [x0, y1 - cap], top, [x1, y0], [x0, y0]]), closed=True)


def graveyard(rates: pd.DataFrame, summary: dict, problems: dict) -> None:
    fig, bg = canvas(["#13314a", "#0a1a2a", "#050b12"])
    stars(bg)
    moon(fig, bg, 136, 76, 11)
    for x, y, s, f in ((128, 90, 3.6, 0.1), (141, 84, 2.4, -0.15), (119, 83, 1.9, 0.2), (150, 92, 1.6, 0)):
        bat(bg, x, y, s, flap=f)
    rng = np.random.default_rng(12)
    dead_tree(bg, 150, 10, 14, np.pi / 2 + 0.15, 8, 9, "#03070b", rng)
    fig._keepout.append(("tree", (140, 10, 160, 52)))
    xs = np.linspace(0, 160, 200)  # rolling hills, far and near
    bg.fill_between(xs, 0, 20 + 4 * np.sin(xs / 19 + 1) + 2 * np.sin(xs / 7), color="#0b1a26", zorder=1.4)
    bg.fill_between(xs, 0, 13 + 2.5 * np.sin(xs / 13), color="#061019", zorder=1.45)
    for x in range(14, 132, 9):  # distant iron fence
        h = 15.6 + 2.5 * np.sin(x / 13)
        bg.plot([x, x], [h - 0.5, h + 4.2], color="#0a1620", lw=2, zorder=1.42)
        bg.add_patch(Polygon([(x - 0.5, h + 4.2), (x + 0.5, h + 4.2), (x, h + 5.3)], color="#0a1620", zorder=1.42))
    bg.plot([14, 131], [17.5, 17.6], color="#0a1620", lw=1.6, zorder=1.42)
    fog(bg, 0, 160, 8, 26, "#9fb8cc", 0.28, 5, 1.6)
    vignette(bg)
    grain(bg)

    r = rates[rates["scope"] == "model"].set_index("group").loc[MODELS]
    allr = rates[rates["scope"] == "overall"].iloc[0]
    vals = [*r["ghost_rate"], allr.ghost_rate]
    lo, hi = [*r["rate_lo"], allr.rate_lo], [*r["rate_hi"], allr.rate_hi]
    counts = [f"{int(a.ghosts)} of {int(a.mentions)}" for a in r.itertuples()] + [f"{int(allr.ghosts)} of {int(allr.mentions):,}"]
    names, glows = MODELS + ["all models"], [MODEL_COLORS[m] for m in MODELS] + ["#d8d2e2"]

    ax = data_axes(fig, [PLOT_X, 0.2, 0.64, 0.52], "#d9d4e4", "#22384c", (-0.55, 3.55), (0, 0.30))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks(range(4), [f"{n}\n{c}" for n, c in zip(names, counts)])
    ticks(ax, "#eee9f5")
    ax.tick_params(axis="x", length=0, pad=14)
    ax.grid(axis="x", visible=False)
    axis_label(ax, "y", "install suggestions that don't exist on PyPI", "#d9d4e4")
    for i, (v, l, h, g) in enumerate(zip(vals, lo, hi, glows)):
        radial_glow(ax, i, 0.0, 0.55, g, alpha=0.55, z=3.5, ry=0.05, power=1.6)  # lantern light on the grass
        cap = min(0.035, v * 0.42)
        x0, x1 = i - 0.27, i + 0.27
        stone = PathPatch(tombstone_path(x0, x1, 0, v, cap), facecolor="none", edgecolor="none", zorder=4)
        ax.add_patch(stone)
        fill_gradient(ax, stone, [x0, x1, 0, v], "#6d6878", "#b4afbb", 4, horizontal=True, noise=0.035, seed=i)
        ax.add_patch(PathPatch(tombstone_path(x0, x1, 0, v, cap), facecolor="none", edgecolor="#1d1a22", lw=2.5,
                               zorder=4.2))
        inner = tombstone_path(x0 + 0.035, x1 - 0.035, 0.01, v - 0.008, cap * 0.85)
        ax.add_patch(PathPatch(inner, facecolor="none", edgecolor="#56515e", lw=1.2, zorder=4.3, alpha=0.8))
        crack = np.c_[np.linspace(x1 - 0.07, x1 - 0.13, 5), np.linspace(v - cap * 0.55, v * 0.68, 5)]  # above R.I.P.
        crack[:, 0] += np.random.default_rng(i).normal(0, 0.012, 5)
        ax.plot(crack[:, 0], crack[:, 1], color="#3d3944", lw=1.4, zorder=4.4)
        for k in range(14):  # moss at the foot
            mx = x0 + 0.02 + k * (x1 - x0 - 0.04) / 13
            ax.add_patch(Ellipse((mx, 0.004), 0.05, 0.010 + 0.006 * (k % 3), color="#2f4a2a", zorder=4.5, lw=0))
        ax.plot([x0 + 0.02, x1 - 0.02], [0.0035, 0.0035], color=g, lw=5, zorder=4.6, alpha=0.95)
        ax.text(i, v * 0.5, "R.I.P.", ha="center", va="center", color="#3e3946", zorder=5, alpha=0.9,
                **fp("georgia", 19 if v > 0.1 else 15))
        ci_whisker(ax, i + 0.36, l, h, "#f4ecd0", 0.04)
        ax.text(i, v + 0.008, f"{v:.1%}", ha="center", va="bottom", color="#fff6e0", zorder=10,
                path_effects=outline(5), **fp("body", 27))
    fog(ax, -0.55, 3.55, 0, 0.03, "#b8cadb", 0.35, 9, 6)
    title(fig, "Where the ghosts are buried", f"Ghost rate by model · {summary['study']['n_generations']:,} "
          "generations of 200 coding tasks · whiskers: 95% bootstrap CI by prompt", "chiller", "#eef4f8",
          "#c9d6df", 82, 0.795)
    footer(fig, "#9fb2c0")
    save(fig, "halloween_ghost_rate_by_model.png", problems)


# ---------------------------------------------------------------- 2. candle ritual

def flame_shape(x, y, w, h):
    t = np.linspace(0, 2 * np.pi, 80)
    px = x + w / 2 * np.sin(t) * (1 - 0.35 * (1 + np.cos(t)) / 2) ** 1.2
    py = y + h * (1 - np.cos(t)) / 2 * (1 + 0.15 * np.sin(t))
    return np.c_[px, py]


def candles(rates: pd.DataFrame, problems: dict) -> None:
    fig, bg = canvas(["#3a0f28", "#1d0714", "#080206"])
    rng = np.random.default_rng(11)
    for _ in range(55):  # embers
        x, y, s = rng.uniform(0, 160), rng.uniform(14, 98), rng.uniform(0.12, 0.45)
        radial_glow(bg, x, y, s * 4, "#ffb347", alpha=rng.uniform(0.15, 0.45), z=1.1)
    cx, cy, rad = 133, 47, 21  # occult circle behind the right column
    for rr, lw in ((rad, 2.2), (rad - 2.4, 1.2)):
        bg.add_patch(Circle((cx, cy), rr, fill=False, edgecolor="#7a2c52", lw=lw, alpha=0.55, zorder=1.2))
    ang = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 2
    pts = np.c_[cx + (rad - 2.4) * np.cos(ang), cy + (rad - 2.4) * np.sin(ang)]
    star = pts[[0, 2, 4, 1, 3, 0]]
    bg.plot(star[:, 0], star[:, 1], color="#7a2c52", lw=1.4, alpha=0.5, zorder=1.2)
    runes = "♈♉♊♋♌♍♎♏♐♑♒♓" * 2  # zodiac glyphs exist in Segoe UI Symbol
    for k, ch in enumerate(runes):
        a = 2 * np.pi * k / len(runes)
        bg.text(cx + (rad - 1.2) * np.cos(a), cy + (rad - 1.2) * np.sin(a), ch, ha="center", va="center",
                rotation=np.degrees(a) - 90, color="#9a3a68", alpha=0.6, gid=DECOR, zorder=1.2,
                fontproperties=font_manager.FontProperties(fname="C:/Windows/Fonts/seguisym.ttf", size=11))
    bg.fill_between([0, 160], 0, 11, color="#12040b", zorder=1.4)  # altar cloth
    bg.plot([0, 160], [11, 11], color="#5a1a34", lw=2, zorder=1.41)
    vignette(bg, 0.6)
    grain(bg)

    r = rates[rates["scope"] == "niche_level"].set_index("group").loc[["L1", "L2", "L3"]]
    names = {"L1": "L1 · everyday", "L2": "L2 · specialised", "L3": "L3 · obscure"}
    ax = data_axes(fig, [PLOT_X, 0.17, 0.6, 0.58], "#ead7c8", "#4a1f36", (-0.6, 2.6), (0, 0.45))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks(range(3), [f"{names[g]}\n{int(x.ghosts)} of {int(x.mentions)}" for g, x in r.iterrows()])
    ticks(ax, "#f6e6d6")
    ax.tick_params(axis="x", length=0, pad=14)
    ax.grid(axis="x", visible=False)
    axis_label(ax, "y", "install suggestions that don't exist on PyPI", "#ead7c8")
    fh = 0.055  # flame height (decoration above the wax line)
    for i, (g, x) in enumerate(r.iterrows()):
        v = x.ghost_rate
        radial_glow(ax, i, v + fh * 0.6, 0.55, "#ffae42", alpha=0.42, z=3.2, ry=0.09, power=1.7)
        dish = Ellipse((i, 0.002), 0.78, 0.024, facecolor="none", edgecolor="none", zorder=3.6)
        ax.add_patch(dish)
        fill_gradient(ax, dish, [i - 0.39, i + 0.39, -0.01, 0.014], "#5a3a14", "#c99a4a", 3.6, horizontal=True,
                      center=0.4)
        body = Polygon([[i - 0.22, 0.004], [i + 0.22, 0.004], [i + 0.22, v], [i - 0.22, v]], closed=True,
                       facecolor="none", edgecolor="none", zorder=4)
        ax.add_patch(body)
        fill_gradient(ax, body, [i - 0.22, i + 0.22, 0, v], "#b99a72", "#fff6e2", 4, center=0.38, noise=0.01, seed=i)
        ax.add_patch(Polygon(body.get_xy(), closed=True, facecolor="none", edgecolor="#6b4a2a", lw=1.6, zorder=4.2))
        for dx, depth in ((-0.15, 0.5), (-0.02, 0.85), (0.12, 0.35), (0.19, 0.6)):  # wax drips
            d = min(depth * v * 0.4, 0.05)
            drip = np.c_[[i + dx - 0.035, i + dx - 0.035, i + dx, i + dx + 0.035, i + dx + 0.035],
                         [v, v - d, v - d - 0.006, v - d, v]]
            ax.add_patch(Polygon(drip, closed=True, facecolor="#fbeed6", edgecolor="none", zorder=4.3))
        ax.add_patch(Ellipse((i, v), 0.44, 0.012, facecolor="#fff2d4", edgecolor="#6b4a2a", lw=1.2, zorder=4.4))
        ax.plot([i, i], [v, v + 0.012], color="#1f120a", lw=2.6, zorder=5)
        for wd, ht, c, a, z in ((0.17, fh, "#ff6a10", 0.6, 5.1), (0.11, fh * 0.75, "#ffb032", 0.95, 5.2),
                                (0.055, fh * 0.42, "#fff4c0", 1.0, 5.3)):
            ax.add_patch(Polygon(flame_shape(i, v + 0.008, wd, ht), closed=True, color=c, alpha=a, zorder=z, lw=0))
        ci_whisker(ax, i + 0.33, x.rate_lo, x.rate_hi, "#ffd18a", 0.035)
        ax.text(i, v + fh + 0.022, f"{v:.1%}", ha="center", va="bottom", color="#fff2dc", zorder=10,
                path_effects=outline(5), **fp("body", 28))
    l1, l3 = r.loc["L1", "ghost_rate"], r.loc["L3", "ghost_rate"]
    panel(fig, 0.75, [(f"{l3 / l1:.0f}×", "onyx", 96, "#ffcf7a", 0.175),
                      ("On obscure tasks the\nmodels invented this\nmany more packages\nthan on everyday ones.",
                       "body_sb", 20, "#f6e3cf", 0.19),
                      (f"everyday: {l1:.1%}\nobscure: {l3:.1%}", "body", 20, "#ffd18a", 0.11),
                      ("gold whiskers: 95% CI\nthe wax line marks the rate;\nflames are decoration", "body_r", 15,
                       "#d9bfae", 0.0)])
    title(fig, "The darker the task, the brighter they burn", "Ghost rate by how niche the task was · all "
          "three models pooled", "onyx", "#ffe2b6", "#ecd2bd", 74, 0.82)
    footer(fig, "#c9a99a")
    save(fig, "halloween_ghost_rate_by_niche.png", problems)


# ---------------------------------------------------------------- 3. spider web

def spider_web(bg, cx, cy, radius, spokes=13, color="#d6d8e4", alpha=0.55, z=1.45, seed=2):
    rng = np.random.default_rng(seed)
    ang = np.sort(rng.uniform(0, 2 * np.pi, spokes) * 0.15 + np.linspace(0, 2 * np.pi, spokes, endpoint=False))
    for a in ang:
        bg.plot([cx, cx + radius * np.cos(a)], [cy, cy + radius * np.sin(a)], color=color, lw=1.0, alpha=alpha, zorder=z)
    for rr in np.arange(2.2, radius, 2.0):
        for a0, a1 in zip(ang, np.r_[ang[1:], ang[0] + 2 * np.pi]):
            t = np.linspace(0, 1, 14)
            am = a0 + (a1 - a0) * t
            sag = rr * (1 - 0.12 * np.sin(np.pi * t))  # threads sag towards the hub
            bg.plot(cx + sag * np.cos(am), cy + sag * np.sin(am), color=color, lw=0.8, alpha=alpha * 0.85, zorder=z)
    for _ in range(22):  # dew drops
        a, rr = rng.choice(ang), rng.uniform(3, radius)
        bg.add_patch(Circle((cx + rr * np.cos(a), cy + rr * np.sin(a)), rng.uniform(0.15, 0.35), color="#ffffff",
                            alpha=0.75, zorder=z + 0.01, lw=0))


def spider(ax, x, y, s, k, body="#ff8a2a", legs="#efe2cc", z=7, thread_top=None):
    """Spider centred at (x, y); s = body size in x units, k = y/x aspect correction."""
    if thread_top is not None:
        ax.plot([x, x], [y + s * 1.6 * k, thread_top], color="#e6e6ee", lw=0.9, alpha=0.55, zorder=z - 2)
    for side in (-1, 1):
        for ang in (0.75, 0.3, -0.15, -0.6):
            knee = (x + side * np.cos(ang) * s * 1.5, y + (np.sin(ang) * s * 1.5 + s * 0.9) * k)
            foot = (x + side * np.cos(ang) * s * 2.6, y + (np.sin(ang) * s * 2.6 - s * 0.6) * k)
            ax.plot([x, knee[0], foot[0]], [y, knee[1], foot[1]], color="#07040a", lw=3.6, zorder=z - 0.2,
                    solid_capstyle="round", solid_joinstyle="round")
            ax.plot([x, knee[0], foot[0]], [y, knee[1], foot[1]], color=legs, lw=1.8, zorder=z,
                    solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(Ellipse((x, y - s * 0.35 * k), s * 1.5, s * 1.8 * k, color=body, ec="#2a1404", lw=1.6, zorder=z + 0.1))
    ax.add_patch(Ellipse((x - s * 0.25, y - s * 0.1 * k), s * 0.45, s * 0.6 * k, color="#ffc07a", alpha=0.7,
                         zorder=z + 0.15, lw=0))
    ax.add_patch(Circle((x, y + s * 0.75 * k), s * 0.5, color=body, ec="#2a1404", lw=1.4, zorder=z + 0.2))


def web(calib: pd.DataFrame, problems: dict) -> None:
    fig, bg = canvas(["#151622", "#0b0b13", "#040407"])
    stars(bg, n=90, ymin=40, color="#dfe4ff", seed=8)
    moon(fig, bg, 143, 80, 11, c0="#f4f5fb", c1="#c9ccdc", glow="#bfc8ff", glow_size=2.6)
    spider_web(bg, 143, 80, 21)
    spider_web(bg, 0, 100, 34, spokes=9, alpha=0.22, seed=5)
    bg.plot([156, 156], [99, 60], color="#e6e6ee", lw=0.9, alpha=0.6, zorder=1.5)
    spider(bg, 156, 58.5, 1.1, 1.0, body="#1a1220", legs="#bdb6c8", z=1.6)
    bat(bg, 118, 92, 2.2)
    fog(bg, 0, 160, 0, 16, "#8a8fb4", 0.25, 3, 1.6)
    vignette(bg)
    grain(bg)

    rect = [PLOT_X, 0.13, 0.52, 0.6]
    ax = data_axes(fig, rect, "#dcdce6", "#262636", (0, 1), (0, 1.04))
    ticks(ax, "#ececf4")
    k = (rect[2] * W) / (rect[3] * H) * (1.04 / 1.0)  # makes a unit circle look round on these axes
    ax.plot([0, 1], [0, 1], color="#e8e8f0", lw=1.6, alpha=0.85, zorder=3, linestyle=(0, (1.5, 3)))
    gb = calib[(calib["arm"] == "hgb") & (calib["n"] >= 5)]
    ax.plot(gb["mean_pred"], gb["frac_ghost"], color="#8f8fac", lw=2, zorder=4, marker="o", ms=7, mec="#0b0b13")
    t = calib[(calib["arm"] == "tabpfn") & (calib["n"] >= 5)]
    ax.plot(t["mean_pred"], t["frac_ghost"], color="#ff8a2a", lw=4.5, zorder=5, path_effects=outline(8, "#2a1404"))
    for row in t.itertuples():
        spider(ax, row.mean_pred, row.frac_ghost, 0.0115, k, thread_top=1.04)
    ax.text(0.97, 0.05, "dotted line: perfect calibration", ha="right", color="#e8e8f0", zorder=8,
            path_effects=outline(4), **fp("body_r", 14))
    axis_label(ax, "x", "TabPFN's predicted P(ghost), 10 equal-width bins", "#dcdce6")
    axis_label(ax, "y", "share that really were ghosts", "#dcdce6")
    ece = {a: float((c["n"] * (c["mean_pred"] - c["frac_ghost"]).abs()).sum() / c["n"].sum())
           for a, c in calib[calib["n"] > 0].groupby("arm")}
    panel(fig, 0.6, [("Each spider is one bin\nof TabPFN predictions.\nOn the thread means\nhonest.", "viner", 24,
                      "#f2f2fa", 0.275),
                     ("calibration error (ECE)", "body_sb", 18, "#c8c8dc", 0.045),
                     (f"TabPFN  {ece['tabpfn']:.3f}", "body", 24, "#ffa45a", 0.05),
                     (f"gradient boosting  {ece['hgb']:.3f}", "body", 19, "#b8b8d0", 0.04),
                     (f"logistic regression  {ece['logreg']:.3f}", "body", 19, "#b8b8d0", 0.06),
                     ("grey line: gradient boosting\nbins with < 5 rows hidden", "body_r", 14, "#a8a8c0", 0.0)], x=0.665)
    title(fig, "Caught in the web of honest odds", "When TabPFN says 80%, about 80% really are ghosts · "
          "out-of-fold, split A (by prompt)", "viner", "#f2f2fa", "#c8c8d8", 52, 0.81)
    footer(fig, "#9a9ab0")
    save(fig, "halloween_calibration.png", problems)


# ---------------------------------------------------------------- 4. ouija board

def planchette(ax, x, y, s, k):
    t = np.linspace(0, 2 * np.pi, 90)
    shape = np.c_[x + s * np.sin(t) * (1 - 0.28 * np.cos(t)), y + s * k * (1.15 * np.cos(t) - 0.1)]
    parts = [Polygon(shape, closed=True, facecolor="#efdcb5", edgecolor="#2b1a10", lw=2.4, zorder=8),
             Ellipse((x, y), s * 0.95, s * 0.95 * k, facecolor="#3b2414", edgecolor="#2b1a10", lw=2, zorder=9),
             Ellipse((x, y), s * 0.7, s * 0.7 * k, facecolor="none", edgecolor="#f3e2c0", lw=1.5, zorder=10)]
    for p in parts:
        p.set_clip_on(False)  # points at recall 1.0 sit on the axis edge
        ax.add_patch(p)


def sun(bg, x, y, r, color="#e9c98a"):
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        bg.add_patch(Polygon([(x + r * 1.15 * np.cos(a - 0.1), y + r * 1.15 * np.sin(a - 0.1)),
                              (x + r * 1.75 * np.cos(a), y + r * 1.75 * np.sin(a)),
                              (x + r * 1.15 * np.cos(a + 0.1), y + r * 1.15 * np.sin(a + 0.1))],
                             color=color, alpha=0.8, zorder=1.6, lw=0))
    bg.add_patch(Circle((x, y), r, color=color, zorder=1.6, lw=0))
    bg.add_patch(Circle((x - r * 0.35, y + r * 0.2), r * 0.12, color="#5a3720", zorder=1.61))
    bg.add_patch(Circle((x + r * 0.35, y + r * 0.2), r * 0.12, color="#5a3720", zorder=1.61))
    bg.plot(x + r * 0.45 * np.cos(np.linspace(3.6, 5.8, 20)), y - r * 0.05 + r * 0.45 * np.sin(np.linspace(3.6, 5.8, 20)),
            color="#5a3720", lw=2, zorder=1.61)


def crescent(bg, x, y, r, color="#e9c98a", board="#4a2c18"):
    bg.add_patch(Circle((x, y), r, color=color, zorder=1.6, lw=0))
    bg.add_patch(Circle((x + r * 0.45, y + r * 0.15), r * 0.9, color=board, zorder=1.61, lw=0))


def ouija(oof: pd.DataFrame, summary: dict, problems: dict) -> None:
    fig, bg = canvas(["#2a160a", "#1a0d05", "#0c0603"])
    board = FancyBboxPatch((2.5, 2.5), 155, 95, boxstyle="round,pad=0,rounding_size=6", facecolor="none",
                           edgecolor="none", zorder=1.1)
    bg.add_patch(board)
    fill_gradient(bg, board, [2.5, 157.5, 2.5, 97.5], "#3a2212", "#6b4426", 1.1, center=0.45, noise=0.02, seed=7)
    rng = np.random.default_rng(5)
    xs = np.linspace(2.5, 157.5, 300)
    for yy in np.linspace(4, 96, 70):  # wood grain
        line = bg.plot(xs, yy + 0.5 * np.sin(xs / rng.uniform(5, 12) + rng.uniform(0, 6)) + 0.3 * np.sin(xs / 2.3),
                       color="#2a170a", alpha=rng.uniform(0.12, 0.3), lw=rng.uniform(0.6, 1.6), zorder=1.15)[0]
        line.set_clip_path(board)
    for pad, lw in ((1.6, 2.2), (2.6, 1.0)):
        bg.add_patch(FancyBboxPatch((2.5 + pad, 2.5 + pad), 155 - 2 * pad, 95 - 2 * pad,
                                    boxstyle="round,pad=0,rounding_size=5", fill=False, edgecolor="#c9a46a",
                                    lw=lw, alpha=0.7, zorder=1.3))
    for k, ch in enumerate("ABCDEFGHIJKLM"):  # two letter arcs between subtitle and chart, like a real board
        a = np.pi * (0.8 - 0.6 * k / 12)
        bg.text(62 + 54 * np.cos(a), 58 + 20 * np.sin(a), ch, color="#e8d2a8", alpha=0.42, ha="center", va="center",
                rotation=np.degrees(a) - 90, gid=DECOR, zorder=1.4, **fp("georgia", 22))
    for k, ch in enumerate("NOPQRSTUVWXYZ"):
        a = np.pi * (0.78 - 0.56 * k / 12)
        bg.text(62 + 48 * np.cos(a), 55 + 16 * np.sin(a), ch, color="#e8d2a8", alpha=0.42, ha="center", va="center",
                rotation=np.degrees(a) - 90, gid=DECOR, zorder=1.4, **fp("georgia", 22))
    sun(bg, 147, 73.5, 3.2)
    crescent(bg, 146, 52, 3.4, board="#5a3820")
    for k, ch in enumerate("1234567890"):
        bg.text(118 + k * 3.6, 7.2, ch, color="#e8d2a8", alpha=0.4, ha="center", gid=DECOR, zorder=1.4,
                **fp("georgia", 20))
    bg.text(127, 12.6, "GOOD BYE", color="#e8d2a8", alpha=0.45, ha="center", gid=DECOR, zorder=1.4, **fp("georgia", 22))
    vignette(bg, 0.5)
    grain(bg, alpha=0.04)

    y = oof["y_ghost"].to_numpy(int)
    rect = [PLOT_X, 0.13, 0.58, 0.5]
    ax = data_axes(fig, rect, "#f0dcb4", "#6a4428", (0, 1.02), (0, 1.03))
    ticks(ax, "#f7e8c8")
    k = (rect[2] * W / 1.02) / (rect[3] * H / 1.03)
    ax.axvline(0.9, color="#f0dcb4", lw=1.4, linestyle=(0, (4, 4)), alpha=0.8, zorder=3)
    ax.text(0.89, 0.03, "90% recall: the guard's target", ha="right", va="bottom", color="#f0dcb4",
            zorder=8, path_effects=outline(3, "#24140a"), **fp("body_r", 13))
    styles = [("rule_editdist", "#a7835a", 2.2, "edit-distance rule"), ("logreg", "#cfae7c", 2.2, "logistic regression"),
              ("hgb", "#f0d6a0", 2.2, "gradient boosting"), ("tabpfn", "#ff9b2e", 4.6, "TabPFN-3.5")]
    for j, (arm, color, lw, label) in enumerate(styles):
        p, r, _ = precision_recall_curve(y, oof[f"A_{arm}"])
        ax.plot(r, p, color=color, lw=lw, drawstyle="steps-post", zorder=5 if arm == "tabpfn" else 4,
                path_effects=outline(lw + 3, "#24140a"))
        yy = 0.31 - 0.07 * (3 - j)
        ax.plot([0.04, 0.1], [yy] * 2, color=color, lw=lw, transform=ax.transAxes, zorder=6,
                path_effects=outline(lw + 3, "#24140a"))
        ax.text(0.12, yy, f"{label}   AUPRC {summary['eval']['A'][arm]['auprc']:.2f}", transform=ax.transAxes,
                va="center", color="#fff1d6", zorder=8, path_effects=outline(4, "#24140a"),
                **fp("body" if arm == "tabpfn" else "body_r", 15))
    for arm, label, at in (("rule_top15k", '"not in top 15k" rule', (0.66, 0.6)),
                           ("llm_selfcheck", "gemma4 asks itself", (0.66, 0.42))):
        flag = oof[f"A_{arm}"].to_numpy() >= 0.5
        tp = int((flag & (y == 1)).sum())
        rec, prec = tp / y.sum(), tp / max(int(flag.sum()), 1)
        planchette(ax, rec, prec, 0.028, k)
        ax.annotate(f"{label} (yes/no)\nrecall {rec:.2f} · precision {prec:.2f}", (rec - 0.035, prec), xytext=at,
                    ha="right", va="center", color="#fff1d6", zorder=8, path_effects=outline(4, "#24140a"),
                    arrowprops={"arrowstyle": "-", "color": "#f0dcb4", "lw": 1.3, "alpha": 0.85}, **fp("body_r", 14))
    axis_label(ax, "x", "recall (share of ghosts caught)", "#f0dcb4")
    axis_label(ax, "y", "precision (share of flags that are ghosts)", "#f0dcb4")
    sc, auprc = summary["selfcheck"], summary["eval"]["A"]["tabpfn"]["auprc"]
    for x0, y0, big, small in ((0.775, 0.735, "YES", f"Is it a ghost? TabPFN\nanswers before PyPI:\nAUPRC {auprc:.2f}."),
                               (0.775, 0.52, "NO", f"gemma4 said NO (doesn't\nexist) to {sc['real_names_said_missing']} of "
                                                  f"{sc['real_names']}\nreal packages.")):
        fig.text(x0, y0, big, color="#f3e2c0", va="center", path_effects=outline(5, "#24140a"), **fp("georgia", 50))
        fig.text(x0, y0 - 0.06, small, color="#ffd9a0", va="top", linespacing=1.3, path_effects=outline(4, "#24140a"),
                 **fp("georgia_r", 18))
    title(fig, "Ask the board: is it a ghost?", "Precision vs. recall before asking PyPI · out-of-fold, split A "
          "(GroupKFold by prompt) · 888 names, 173 ghosts", "georgia", "#fbe9c6", "#ead2a8", 54, 0.835)
    footer(fig, "#d8bf96")
    save(fig, "halloween_precision_recall.png", problems)


# ---------------------------------------------------------------- 5. pumpkin patch

def ghost_shape(x, h, w):
    cap = min(h * 0.45, 10)
    t = np.linspace(np.pi, 0, 50)
    top = np.c_[x + w / 2 * np.cos(t), (h - cap) + cap * np.sin(t)]
    hem_x = np.linspace(x + w / 2, x - w / 2, 41)
    hem = np.c_[hem_x, 0.6 * np.abs(np.sin((hem_x - x) / w * 3 * np.pi)) - 0.2]
    return np.vstack([[x - w / 2, h - cap], top, [x + w / 2, 0.4], hem]), cap


def ghost(ax, x, h, w, c_top, c_bottom, k, z=4):
    if h <= 0:
        return
    pts, cap = ghost_shape(x, h, w)
    radial_glow(ax, x, h * 0.55, w * 1.1, c_top, alpha=0.22, z=z - 0.5, ry=h * 0.7 + 4, power=1.5)
    body = Polygon(pts, closed=True, facecolor="none", edgecolor="none", zorder=z)
    ax.add_patch(body)
    fill_gradient(ax, body, [x - w / 2, x + w / 2, -0.5, h], c_bottom, c_top, z)
    ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor="#2a1206", lw=2, zorder=z + 0.1))
    if h > 8:
        ey = h - cap * 0.7
        for dx in (-w * 0.17, w * 0.17):
            ax.add_patch(Ellipse((x + dx, ey), w * 0.12, w * 0.17 * k, color="#1a0c05", zorder=z + 0.2))
            ax.add_patch(Ellipse((x + dx + w * 0.02, ey + w * 0.04 * k), w * 0.035, w * 0.05 * k, color="#ffffff",
                                 zorder=z + 0.3))
        ax.add_patch(Ellipse((x, ey - w * 0.2 * k), w * 0.1, w * 0.13 * k, color="#1a0c05", zorder=z + 0.2))
        for dx in (-w * 0.3, w * 0.3):  # blush
            ax.add_patch(Ellipse((x + dx, ey - w * 0.12 * k), w * 0.1, w * 0.05 * k, color="#ff9a9a", alpha=0.45,
                                 zorder=z + 0.2, lw=0))


def pumpkin(bg, x, y, s, z=2.0):
    radial_glow(bg, x, y, s * 2.2, "#ff9a2a", alpha=0.35, z=z - 0.1)
    for dx, c in ((-0.55, "#a8420c"), (0.55, "#a8420c"), (-0.28, "#d85a14"), (0.28, "#d85a14"), (0, "#ec7a22")):
        bg.add_patch(Ellipse((x + dx * s, y), s * 0.95, s * 1.15, color=c, zorder=z, ec="#6a2606", lw=1))
    bg.add_patch(Polygon([[x - 0.1 * s, y + 0.5 * s], [x + 0.12 * s, y + 0.5 * s], [x + 0.25 * s, y + 0.9 * s],
                          [x + 0.02 * s, y + 0.82 * s]], color="#3b5a1f", zorder=z))
    glow = "#ffd23a"
    for dx in (-0.36, 0.36):
        bg.add_patch(Polygon([[x + dx * s - 0.16 * s, y + 0.06 * s], [x + dx * s + 0.16 * s, y + 0.06 * s],
                              [x + dx * s, y + 0.32 * s]], color=glow, zorder=z + 0.1))
    mouth = [[x - 0.5 * s, y - 0.14 * s], [x - 0.32 * s, y - 0.3 * s], [x - 0.16 * s, y - 0.2 * s],
             [x, y - 0.36 * s], [x + 0.16 * s, y - 0.2 * s], [x + 0.32 * s, y - 0.3 * s], [x + 0.5 * s, y - 0.14 * s],
             [x + 0.3 * s, y - 0.44 * s], [x, y - 0.52 * s], [x - 0.3 * s, y - 0.44 * s]]
    bg.add_patch(Polygon(mouth, color=glow, zorder=z + 0.1))


def haunted_house(fig, bg, x, y, s):
    c = "#07030a"
    hill = np.c_[np.linspace(x - 12 * s, x + 14 * s, 40), y + 1.2 * s * np.sin(np.linspace(0, np.pi, 40))]
    bg.add_patch(Polygon(np.vstack([[x - 12 * s, 0], hill, [x + 14 * s, 0]]), color=c, zorder=1.5))
    parts = [[(x - 5, y), (x - 5, y + 7), (x - 6, y + 7), (x - 2.5, y + 11), (x + 1, y + 7), (x, y + 7), (x, y)],
             [(x, y), (x, y + 5), (x - 0.8, y + 5), (x + 3.5, y + 9), (x + 7.8, y + 5), (x + 7, y + 5), (x + 7, y)],
             [(x + 7, y), (x + 7, y + 10), (x + 6.3, y + 10), (x + 8.3, y + 14.5), (x + 10.3, y + 10),
              (x + 9.6, y + 10), (x + 9.6, y)]]
    for poly in parts:
        bg.add_patch(Polygon([(px * 1, py) for px, py in poly], color=c, zorder=1.55))
    for wx, wy in ((x - 3.6, y + 4.2), (x - 1.6, y + 4.2), (x + 2.2, y + 2.4), (x + 4.6, y + 2.4), (x + 7.8, y + 7.4)):
        radial_glow(bg, wx + 0.45, wy + 0.6, 1.8, "#ffcf4a", alpha=0.5, z=1.56)
        bg.add_patch(Polygon([(wx, wy), (wx + 0.9, wy), (wx + 0.9, wy + 1.2), (wx, wy + 1.2)], color="#ffd65a", zorder=1.57))
    fig._keepout.append(("house", (x - 6, y, x + 10.5, y + 14.5)))


def pumpkin_patch(counts: pd.DataFrame, summary: dict, problems: dict) -> None:
    fig, bg = canvas(["#3a1606", "#1e0a08", "#080305"])
    stars(bg, n=110, ymin=58, color="#ffe2b0", seed=12)
    moon(fig, bg, 147, 88, 7.5, c0="#ffe3ad", c1="#e7a65a", glow="#ff9a40", glow_size=3.0)
    for x, y, s, f in ((122, 92, 2.6, 0.15), (131, 86, 1.7, -0.1), (110, 95, 1.3, 0.1)):
        bat(bg, x, y, s, flap=f)
    haunted_house(fig, bg, 140, 13, 1.0)
    bg.fill_between([0, 160], 0, 8, color="#0d0503", zorder=1.6)
    vx = np.linspace(0, 160, 300)
    bg.plot(vx, 8.6 + 0.9 * np.sin(vx / 3.5), color="#2f4a17", lw=2.4, zorder=1.65)
    for cx in (12, 34, 58, 96, 118):  # vine curls
        t = np.linspace(0, 4 * np.pi, 60)
        bg.plot(cx + 0.18 * t * np.cos(t), 9.5 + 0.18 * t * np.sin(t), color="#3d5e1d", lw=1.4, zorder=1.65)
    for x, s in ((7, 4.4), (19, 3.2), (153, 3.4)):
        pumpkin(bg, x, 6.8 + s * 0.45, s)
    fog(bg, 0, 160, 4, 20, "#d6a27a", 0.22, 6, 1.7)
    vignette(bg)
    grain(bg)

    by = counts.set_index("mentions")
    xs, total, multi = by.index.to_numpy(), by["ghost_names"].to_numpy(), by["by_2plus_models"].to_numpy()
    rect = [PLOT_X, 0.2, 0.62, 0.55]
    ax = data_axes(fig, rect, "#f6dcc0", "#43200e", (0.4, xs.max() + 0.6), (-2, total.max() * 1.2))
    k = (rect[2] * W / (xs.max() + 0.2)) / (rect[3] * H / (total.max() * 1.2 + 2))
    ticks(ax, "#fde8d0")
    ax.set_xticks(xs, [f"{i}×" for i in xs])
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=12)
    for x, n, m in zip(xs, total, multi):
        if m == n:
            ghost(ax, x, n, 0.6, "#ffb36a", "#e2620e", k)
        else:
            ghost(ax, x, n, 0.6, "#ffffff", "#cfc7e4", k)
            if m:
                ax.add_patch(Polygon([[x - 0.3, 0.4], [x + 0.3, 0.4], [x + 0.3, m], [x - 0.3, m]], closed=True,
                                     facecolor="#f07a1e", edgecolor="#2a1206", lw=1.6, zorder=4.5))
        ax.text(x, n + total.max() * 0.035, f"{n}", ha="center", va="bottom", color="#fff3e0", zorder=10,
                path_effects=outline(5), **fp("body", 27))
    axis_label(ax, "x", "times the same ghost name was suggested", "#f6dcc0")
    axis_label(ax, "y", "ghost names", "#f6dcc0")
    rep = summary["repeat"]
    panel(fig, 0.72, [(f"{rep['repeated_2plus']} of {rep['n_unique_ghost_names']}", "jokerman", 44, "#ffb347", 0.1),
                      ("ghosts came back", "jokerman", 28, "#ffb347", 0.1),
                      ("white: suggested by one model\norange: by two or more models", "body_sb", 16, "#fde8d0", 0.1),
                      ("Every repeat came from the\nsame task. A predictable\nghost is a name worth\nsquatting.",
                       "body_r", 16, "#f6dcc0", 0.0)])
    title(fig, "The ghosts that keep coming back", f"{rep['n_unique_ghost_names']} unregistered ghost names by "
          "how often a model suggested them · names withheld", "jokerman", "#ffcf7a", "#f0d2b4", 54, 0.825)
    footer(fig, "#c8a07a")
    save(fig, "halloween_repeat_ghosts.png", problems)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_known_args()
    OUT.mkdir(exist_ok=True)
    summary = json.loads(config.SUMMARY_PATH.read_text(encoding="utf-8"))
    rates = pd.read_csv(config.RESULTS / "rates.csv")
    problems: dict[str, list[str]] = {}
    graveyard(rates, summary, problems)
    candles(rates, problems)
    web(pd.read_csv(config.RESULTS / "calibration.csv"), problems)
    ouija(pd.read_parquet(config.DATA / "oof_predictions.parquet"), summary, problems)
    pumpkin_patch(pd.read_csv(config.RESULTS / "repeat_counts.csv"), summary, problems)
    bad = {k: v for k, v in problems.items() if v}
    if bad:
        raise SystemExit(f"layout validation failed: {sum(map(len, bad.values()))} issue(s) in {list(bad)}")
    print("all figures passed layout validation")


if __name__ == "__main__":
    main()
