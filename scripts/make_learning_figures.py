"""Redraw the main paper's routing-performance figure in the house style.

One figure, emitted twice per width (hard project convention):

    <stem>.pdf   light theme, for the paper
    <stem>.png   TRANSPARENT background, light ink, for dark-background slides

    learning_and_entropy_combined        3.3 in, single column
    learning_and_entropy_combined_wide   7.0 in, for \\begin{figure*}

Layout, 2 rows x 3 columns, one column per scenario:
    top    mean travel time (min) against training episode, mean over seeds
           with a +/- 1 sd band, exactly as the notebook's seaborn lineplot
    bottom final mean travel time against global path choice entropy, one
           marker per policy

This REPLACES `graphs/notebooks/learning_graphs.ipynb` cell 3 as the producer
of the figure. The notebook drew at 7.0 in with ~9 pt type and the paper then
scaled it to 0.78\\columnwidth (x0.37), so its labels landed on the page at
about 3.5 pt while Figures 1-3 were set at 7-8 pt. Everything here is drawn at
its FINAL printed size instead.

NOTHING about the plotted values changes. Same CSVs, same seed averaging, same
seconds-to-minutes conversion, same exclusion of `evo_swap` (never reported in
the paper). Only colour, marker, dash, font, grid, spines, legend and figure
size are new. `--check` prints the final-episode means against Table 4.

Usage:
    python scripts/make_learning_figures.py \\
        --learning-dir graphs/data/learning --entropy-dir graphs/data/entropy \\
        --out-dir aamas_2027_template/Figures
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from tamarl.visualisation.ablation_figures import (
    AGENT_COLORS,
    AGENT_COLORS_DARK,
    AGENT_MARKERS,
    finish,
)
from tamarl.visualisation.validation_figures import (
    house_axes,
    house_legend,
    house_palette,
    setup_style,
)

# ---------------------------------------------------------------------------
# Identity. Colour and marker are the shared per-algorithm tables in
# `tamarl/visualisation/ablation_figures.py`, so UCB is the same purple here
# and in the appendix sweeps. Only the dash pattern is added locally: eight
# lines is twice what the appendix figures carry, and colour alone does not
# separate them in a greyscale print or for a colour-blind reader.
#
# Labels are Table 4's own names, not `AGENT_LABELS`' long forms. At single
# column width "Replicator Dynamics" and "Thompson Sampling" force the legend
# onto four rows or down to 5 pt type; the table already calls them RD and TS.
# ---------------------------------------------------------------------------
AGENT_ORDER = ("random", "aon", "msa", "rd", "epsilon_greedy", "exp3", "ucb", "ts")

LABELS = {
    "random": "Random",
    "aon": "AoN",
    "msa": "MSA",
    "rd": "RD",
    "epsilon_greedy": r"$\epsilon$-Greedy",
    "exp3": "Exp3",
    "ucb": "UCB",
    "ts": "TS",
}

# (linestyle, linewidth, zorder). UCB and TS are the paper's result, so they
# ride on top as solid lines; the baselines take the dashes.
STROKE = {
    "random": {"ls": (0, (1, 1.6)), "lw": 1.0, "z": 3},
    "aon": {"ls": (0, (5, 1.6)), "lw": 1.0, "z": 3},
    "msa": {"ls": (0, (3.5, 1.4)), "lw": 1.0, "z": 4},
    "rd": {"ls": (0, (1.2, 1.2, 4, 1.2)), "lw": 1.0, "z": 3},
    "epsilon_greedy": {"ls": (0, (2.2, 1.2)), "lw": 1.1, "z": 4},
    "exp3": {"ls": (0, (4, 1.2, 1, 1.2)), "lw": 1.1, "z": 4},
    "ucb": {"ls": "-", "lw": 1.3, "z": 6},
    "ts": {"ls": "-", "lw": 1.3, "z": 6},
}

SCENARIOS = (
    ("sioux_falls", "Sioux Falls", "entropy_SF.csv"),
    ("berlin", "Berlin", "entropy_Berlin.csv"),
    ("los_angeles", "Los Angeles", "entropy_LA.csv"),
)

Y_METRIC = "metrics/Mean TT"

# Dropped before plotting, as in the notebook: `evo_swap` is not one of the
# eight policies the paper reports, and it is absent from Table 4.
EXCLUDE = {"evo_swap"}


def _norm(agent: str) -> str:
    """`epsilon-greedy` in the entropy CSVs, `epsilon_greedy` in the curves."""
    return agent.replace("-", "_")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_curves(path: str) -> dict:
    """Return {agent: (episodes, mean_minutes, sd_minutes, n_seeds)}.

    Mean and sd are taken ACROSS SEEDS at each episode, which is what
    seaborn's `errorbar="sd"` drew in the notebook. Seconds to minutes, one
    division, same as before.
    """
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        a = _norm(r["agent"])
        v = r.get(Y_METRIC)
        if a in EXCLUDE or not v or v.lower() == "nan" or not r.get("episode"):
            continue
        by[a][int(float(r["episode"]))].append(float(v) / 60.0)

    out = {}
    for a, per_ep in by.items():
        eps = sorted(per_ep)
        out[a] = (
            np.array(eps, dtype=float),
            np.array([st.mean(per_ep[e]) for e in eps]),
            np.array([st.stdev(per_ep[e]) if len(per_ep[e]) > 1 else 0.0
                      for e in eps]),
            np.array([len(per_ep[e]) for e in eps], dtype=int),
        )
    return out


def load_entropy(path: str) -> dict:
    """Return {agent: (mean_entropy, mean_tt_minutes)}."""
    out = {}
    for r in csv.DictReader(open(path)):
        a = _norm(r["algo"])
        if a in EXCLUDE or not r.get("mean_entropy"):
            continue
        out[a] = (float(r["mean_entropy"]), float(r["mean_tt_seconds"]) / 60.0)
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _color(agent: str, dark: bool) -> str:
    return (AGENT_COLORS_DARK if dark else AGENT_COLORS)[agent]


def draw_curves(ax, curves, pal, dark, *, narrow: bool, ylabel=None, title=None):
    """One learning-curve panel: mean over seeds, +/- 1 sd band."""
    n_mark = 5 if narrow else 7
    for i, a in enumerate(AGENT_ORDER):
        if a not in curves:
            continue
        x, y, e, _n = curves[a]
        c = _color(a, dark)
        s = STROKE[a]
        if np.any(e > 0):
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.18, lw=0,
                            zorder=s["z"] - 2)
        # Markers on a 100-point curve would be a solid band, so they are
        # sparse and offset per series: they are legend keys and a greyscale
        # fallback, not data points.
        step = max(1, len(x) // n_mark)
        ax.plot(x, y, color=c, ls=s["ls"], lw=s["lw"], zorder=s["z"],
                marker=AGENT_MARKERS[a],
                ms=(4.2 if AGENT_MARKERS[a] in ("*", "P") else 2.8)
                   * (0.88 if narrow else 1.0),
                markevery=(3 + (i * step) // len(AGENT_ORDER), step),
                mfc=c, mec=c, mew=0.6)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100] if narrow else [0, 25, 50, 75, 100])
    ax.set_xlabel("Episode")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7.5 if not narrow else 7.0,
                     color=pal["muted"], pad=3)
    house_axes(ax, pal, x_grid=True, y_grid=True)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4 if narrow else 5,
                                                   steps=[1, 2, 2.5, 5, 10],
                                                   min_n_ticks=4))


def draw_entropy(ax, pts, pal, dark, *, narrow: bool, ylabel=None):
    """One trade-off panel: final travel time against path choice entropy."""
    for a in AGENT_ORDER:
        if a not in pts:
            continue
        h, tt = pts[a]
        big = a in ("ucb", "ts")
        ax.scatter(h, tt, color=_color(a, dark), marker=AGENT_MARKERS[a],
                   s=(60 if big else 34) * (0.8 if narrow else 1.0),
                   edgecolors=pal["ink"], linewidths=0.4, alpha=0.92,
                   zorder=6 if big else 4, clip_on=False)
    ax.set_xlabel(r"$H_{\mathrm{global}}$" if narrow
                  else r"Path Choice Entropy ($H_{\mathrm{global}}$)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.margins(x=0.09, y=0.11)
    house_axes(ax, pal, x_grid=True, y_grid=True)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3 if narrow else 5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4 if narrow else 5,
                                                   steps=[1, 2, 2.5, 5, 10],
                                                   min_n_ticks=4))


def legend_handles(dark):
    return [
        mlines.Line2D([], [], color=_color(a, dark), ls=STROKE[a]["ls"],
                      lw=STROKE[a]["lw"] + 0.2, marker=AGENT_MARKERS[a],
                      ms=5.0 if AGENT_MARKERS[a] in ("*", "P") else 3.6,
                      mfc=_color(a, dark), mec=_color(a, dark), mew=0.6,
                      label=LABELS[a])
        for a in AGENT_ORDER
    ]


def build(curves, entropy, out_stem, *, narrow: bool):
    """Both themes of one width."""
    figsize = (3.3, 3.95) if narrow else (7.0, 3.85)
    for dark in (False, True):
        pal = house_palette(dark)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        for i, (key, title, _ent) in enumerate(SCENARIOS):
            draw_curves(axes[0, i], curves[key], pal, dark, narrow=narrow,
                        ylabel="Mean Travel Time (min)" if i == 0 else None,
                        title=title)
            draw_entropy(axes[1, i], entropy[key], pal, dark, narrow=narrow,
                         ylabel="Mean Travel Time (min)" if i == 0 else None)

        # Legend above everything, frameless, spanning the width: the layout
        # Figures 2 and 3 use. Eight entries never fit one row at 3.3 in, so
        # the narrow version takes two rows of four.
        # The legend sits in the strip `rect` leaves free at the top. Two rows
        # of 6.4 pt type need about 0.26 in of the narrow figure's 3.95 in.
        top = 0.925 if narrow else 0.945
        fig.tight_layout(pad=0.3, w_pad=0.9, h_pad=1.15, rect=(0, 0, 1, top))
        fig.align_ylabels(axes)
        fig.align_xlabels(axes)
        house_legend(fig, legend_handles(dark), pal,
                     fontsize=6.4 if narrow else 7.6,
                     ncol=4 if narrow else 8, loc="upper center",
                     bbox_to_anchor=(0.5, 1.004),
                     handlelength=1.9 if narrow else 2.0,
                     handletextpad=0.4,
                     columnspacing=1.1 if narrow else 1.3,
                     labelspacing=0.35)
        print(f"  wrote {finish(fig, out_stem, dark)}")


# ---------------------------------------------------------------------------
# Numeric check against Table 4
# ---------------------------------------------------------------------------
# Final routing performance at episode 100, mean travel time in minutes, read
# straight out of `aamas_2027_template/AAMAS_2027_TrafficGym.tex`,
# `tab:performance_results`. The figure's curves must land on these.
TABLE4 = {
    "sioux_falls": {"random": 84.2, "aon": 31.7, "msa": 29.5, "rd": 50.0,
                    "epsilon_greedy": 46.9, "exp3": 59.3, "ucb": 24.1,
                    "ts": 30.5},
    "berlin": {"random": 122.9, "aon": 157.0, "msa": 108.3, "rd": 123.8,
               "epsilon_greedy": 61.1, "exp3": 81.4, "ucb": 25.6, "ts": 25.2},
    "los_angeles": {"random": 114.1, "aon": 118.9, "msa": 93.7, "rd": 107.6,
                    "epsilon_greedy": 55.1, "exp3": 93.1, "ucb": 25.8,
                    "ts": 24.6},
}


def check(curves, tol: float = 0.06) -> bool:
    """Compare each policy's last-episode mean with Table 4. Rounding to one
    decimal is all the table carries, so the tolerance is half a least digit
    plus a little."""
    ok = True
    print("\nfinal-episode mean travel time (min), plotted vs Table 4:")
    for key, title, _ in SCENARIOS:
        print(f"  {title}")
        for a in AGENT_ORDER:
            if a not in curves[key]:
                print(f"    {LABELS[a]:<12} MISSING from the data")
                ok = False
                continue
            x, y, _e, n = curves[key][a]
            got, want = y[-1], TABLE4[key][a]
            bad = abs(got - want) > tol
            ok &= not bad
            print(f"    {a:<15} ep{int(x[-1]):>3}  {got:7.2f}  table {want:6.1f}"
                  f"  d={got - want:+.2f}  seeds={n[-1]}"
                  f"{'   <-- MISMATCH' if bad else ''}")
    print("  all policies match Table 4" if ok else "  MISMATCH, see above")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--learning-dir", default="graphs/data/learning")
    p.add_argument("--entropy-dir", default="graphs/data/entropy")
    p.add_argument("--out-dir", default="aamas_2027_template/Figures")
    p.add_argument("--check-only", action="store_true",
                   help="verify the numbers against Table 4 and draw nothing")
    a = p.parse_args()

    curves, entropy = {}, {}
    for key, _title, ent in SCENARIOS:
        curves[key] = load_curves(f"{a.learning_dir}/{key}_learning_curves.csv")
        entropy[key] = load_entropy(f"{a.entropy_dir}/{ent}")

    if not check(curves):
        print("\nABORT: the curves disagree with Table 4; nothing was written.")
        return 1
    if a.check_only:
        return 0

    usetex = setup_style()
    if not usetex:
        print("note: text.usetex unavailable, STIX serif fallback in use")
    os.makedirs(a.out_dir, exist_ok=True)
    print("\nrendering:")
    build(curves, entropy, os.path.join(a.out_dir,
                                        "learning_and_entropy_combined"),
          narrow=True)
    build(curves, entropy, os.path.join(a.out_dir,
                                        "learning_and_entropy_combined_wide"),
          narrow=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
