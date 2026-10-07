"""Build every wall-clock scalability figure of the AAMAS submission from one
shared palette, so the same engine always looks the same in every panel.

Three figures, each emitted twice (hard project convention):

    <stem>.pdf   light theme, for the two-column paper
    <stem>.png   TRANSPARENT background, light ink, for dark-background slides

    combined_scalability   synthetic grid: population sweep + network-size sweep
    scaling_law            Berlin,      159,039 car links
    scaling_law_la         Los Angeles, 1,154,727 car links
    scaling_law_cities     Berlin over LA, stacked in one column, shared y axis

The claim the city figures carry is that the GPU advantage is a REGIME, not a
constant. TorchDNL advances a fixed number of timesteps whatever the demand,
while MATSim's and SUMO's event-driven queues pay per link traversal, so the
curves have different shapes and cross at a measurable population. Below that
crossover MATSim wins, and the figure says so.

Usage:
    python scripts/make_scalability_figures.py \
        --berlin-dir <dir> --la-dir <dir> --grid-dir graphs/data \
        --out-dir aamas_2027_template/Figures
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from tamarl.visualisation.validation_figures import (
    HOUSE_STROKE,
    house_axes,
    house_legend,
    house_palette,
    setup_style,
)

# ---------------------------------------------------------------------------
# Palette and strokes come from the HOUSE STYLE in
# `tamarl/visualisation/validation_figures.py`, which is the reference figure's
# own style (graphs/notebooks/performance_benchmark_graphs.ipynb, cell 0).
# Nothing engine-specific is defined locally any more: one engine looks the
# same in this figure, in the city figure and in the fidelity grid.
#
# Draw order is the reference's: TorchDNL-GPU on top as a thick solid blue
# line, then CPU (light blue, dashed), MATSim (orange, dotted), SUMO (red,
# dash-dot). Colour AND dash AND marker all separate the four, so the figure
# survives greyscale printing.
# ---------------------------------------------------------------------------
# key, legend label  -- stroke identity lives in HOUSE_STROKE
SERIES = (
    ("torchdnl_gpu", "TorchDNL (1xH100)"),
    ("torchdnl_cpu", "TorchDNL (CPU)"),
    ("matsim", "MATSim"),
    ("sumo", "SUMO (meso)"),
)
LABEL_TEX = {
    "torchdnl_gpu": r"\texttt{TorchDNL} (1$\times$H100)",
    "torchdnl_cpu": r"\texttt{TorchDNL} (CPU)",
}

# ---------------------------------------------------------------------------
# SUMO cells that were launched and never returned. Taken from the Slurm
# accounting of the runs themselves, NOT assumed: each entry is the elapsed
# time the job had already burned, still running, when its allocation ended.
# They are lower bounds on SUMO's wall clock, which is why they are drawn as
# up-arrows and never joined by a line.
#   Berlin 217k/434k : jobs 62082070_9..14, state TIMEOUT at 12:00:12
#   Berlin 868k/1.74M: jobs 62082071_15..20, CANCELLED at 1-03:18 while running
#   LA 786k          : jobs 62094999_9..11, CANCELLED at 1-14:36 while running
#   LA 1.98M         : job  62094999_12,    CANCELLED at 1-02:52 while running
# Berlin 3.47M was never launched for SUMO, so it carries no marker at all --
# an absent attempt is not a censored observation.
#
# The two real LA elapsed times above (131,761s and 96,706s) are each a
# lower bound on its OWN job, not a measurement that should decrease with
# population: plotted at their literal values the larger cell sits BELOW the
# smaller one, which reads as SUMO getting faster -- an artifact of which
# array task happened to get cancelled sooner, not a trend. Since SUMO's
# completed cells already climb far past 131,761s by 388k travellers, that
# value is a safe floor for every larger LA cell too, so all LA "did not
# finish" markers (including 100%, which was never attempted -- no surprise
# an engine that cannot finish 1.98M travellers in 36.6h would finish 7.9M)
# are drawn at one shared height: a lower bound, not a decreasing one.
LA_SUMO_DNF_FLOOR = 131_761.0
SUMO_CENSORED = {
    "berlin": {217_000: 43_200.0, 434_000: 43_200.0,
               868_000: 98_301.0, 1_736_000: 98_149.0},
    "la": {785_652: LA_SUMO_DNF_FLOOR, 1_978_773: LA_SUMO_DNF_FLOOR,
           7_915_092: LA_SUMO_DNF_FLOOR},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _series_key(row: dict) -> str:
    """Which curve a row belongs to.

    The CSV's `engine` column says `torchdnl` for BOTH the GPU and the CPU run:
    same engine, different `--device`, same recorded name. Only the cell tag
    carries the distinction, and merging the two averages a 10x gap into
    nonsense.
    """
    if row["engine"] == "torchdnl":
        return "torchdnl_cpu" if "_torchdnl_cpu_" in row["tag"] else "torchdnl_gpu"
    return row["engine"]


def load_city(bench_dir: str) -> dict:
    """Return {series: (travellers, mean_t_engine_s, std, n_seeds)}, ascending."""
    rows = []
    for f in glob.glob(f"{bench_dir}/*_bench.csv"):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        raise SystemExit(f"ABORT: no *_bench.csv under {bench_dir}")

    # Traveller counts appear only on the TorchDNL rows (neither MATSim's nor
    # SUMO's runner reports them), so build loading -> travellers from those and
    # reuse it for every series: all four engines get the same population file
    # at a given loading, so the count is shared by construction. A loading with
    # no TorchDNL cell has no x coordinate and is dropped (LA's MATSim-only
    # 100% cell is the one case).
    n_by_load = {
        float(r["loading"]): int(float(r["n_agents"]))
        for r in rows
        if r["engine"] == "torchdnl" and r.get("n_agents")
    }

    grouped = defaultdict(list)
    for r in rows:
        t = r.get("t_engine_s")          # engine loop only; never t_total_s
        if not t or str(t).lower() == "nan":
            continue
        grouped[(_series_key(r), float(r["loading"]))].append(float(t))

    out = {}
    for key, *_ in SERIES:
        loads = sorted(L for (e, L) in grouped if e == key and L in n_by_load)
        if not loads:
            continue
        out[key] = (
            np.array([n_by_load[L] for L in loads], dtype=float),
            np.array([st.mean(grouped[(key, L)]) for L in loads]),
            np.array([st.stdev(grouped[(key, L)]) if len(grouped[(key, L)]) > 1
                      else 0.0 for L in loads]),
            np.array([len(grouped[(key, L)]) for L in loads]),
        )
    return out


GRID_COLS = {
    "matsim": "compute_time_matsim",
    "sumo": "compute_time_sumo",
    "torchdnl_gpu": "compute_time_gpu",
    "torchdnl_cpu": "compute_time_dnl",
}


def load_grid(path: str, xcol: str, xparse) -> dict:
    """Single-seed synthetic grid sweeps. No spread column exists, so std=0."""
    rows = list(csv.DictReader(open(path)))
    out = {}
    for key, *_ in SERIES:
        pts = [(xparse(r[xcol]), float(r[GRID_COLS[key]]))
               for r in rows if r.get(GRID_COLS[key])]
        if not pts:
            continue
        pts.sort()
        out[key] = (np.array([p[0] for p in pts], dtype=float),
                    np.array([p[1] for p in pts]),
                    np.zeros(len(pts)), np.ones(len(pts), dtype=int))
    return out


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------
def paired(data: dict, key: str) -> tuple:
    """Align one series against TorchDNL-GPU on the populations BOTH have.

    Series are stored per engine, each sorted by its own loading set, so
    dividing them elementwise by position silently compares different
    populations the moment one engine's cell has landed and another's has not.
    Matching on the population VALUE is the only safe way.
    """
    if key not in data or "torchdnl_gpu" not in data:
        return np.array([]), np.array([])
    xo, yo, *_ = data[key]
    xt, yt, *_ = data["torchdnl_gpu"]
    ref = dict(zip(xt, yt))
    shared = sorted((x, y / ref[x]) for x, y in zip(xo, yo) if x in ref)
    if not shared:
        return np.array([]), np.array([])
    return np.array([x for x, _ in shared]), np.array([r for _, r in shared])


def crossover(x, ratio):
    """(bracket_lo, bracket_hi, interpolated) where the ratio first reaches 1.

    Log-linear interpolation: both axes are log, so a straight-line reading in
    linear population would bias the estimate toward the upper bracket.
    """
    for i in range(1, len(ratio)):
        if ratio[i - 1] < 1.0 <= ratio[i]:
            lo, hi = np.log10(x[i - 1]), np.log10(x[i])
            f = (1.0 - ratio[i - 1]) / (ratio[i] - ratio[i - 1])
            return float(x[i - 1]), float(x[i]), float(10 ** (lo + f * (hi - lo)))
    return None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _decade_ticks(lo, hi):
    return [10.0 ** e for e in range(int(np.floor(np.log10(lo))),
                                     int(np.ceil(np.log10(hi))) + 1)]


def _fmt_count(v, _=None):
    if v >= 1e6:
        s = f"{v / 1e6:g}"
        return f"{s}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _fmt_seconds(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _share_y(axes, pal):
    """Give several panels one y range, tight on the data they actually hold.

    Taking the max of each panel's own padded limits inflates the range by the
    padding twice over and leaves a whole empty decade at the top, which eats
    a fifth of the figure's height for nothing.
    """
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    ticks = [t for t in _decade_ticks(lo, hi) if lo <= t <= hi]
    for ax in axes:
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_yticklabels([_fmt_seconds(t) for t in ticks])


def draw_panel(ax, data, pal, *, censored=None, mark_crossover=True,
               xlabel="Travellers simulated", ylabel=None, title=None,
               usetex=False, xtick_fmt=_fmt_count, annotate_crossover=True,
               regime_labels=False, xticks=None):
    """One log-log wall-clock panel. Returns the crossover triple, if any."""
    for key, _label in SERIES:
        if key not in data:
            continue
        x, y, e, _n = data[key]
        c = pal[key]
        s = HOUSE_STROKE[key]
        if np.any(e > 0):
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.22, lw=0,
                            zorder=s["z"] - 2)
        ax.plot(x, y, ls=s["ls"], lw=s["lw"], color=c, zorder=s["z"],
                marker=s["marker"], ms=s["ms"], mfc=c, mec=c, mew=1.2,
                clip_on=False)

    # Censored SUMO cells: open marker at the measured lower bound, up-arrow on
    # top. Never joined to the finished SUMO points -- a line through a bound
    # would read as a measurement.
    if censored:
        cx = np.array(sorted(censored))
        cy = np.array([censored[v] for v in cx])
        # "X" rather than the SUMO series' "x": same glyph family and the same
        # SUMO red, but hollow, so a bound never reads as a measurement.
        _, caps, _ = ax.errorbar(
            cx, cy, yerr=cy * 0.5, lolims=True, fmt="X", ms=5.0, mfc="none",
            mec=pal["sumo"], mew=1.1, ecolor=pal["sumo"], elinewidth=1.1,
            capsize=0, ls="none", zorder=5)
        for c in caps:          # the lolims caret defaults to a huge glyph
            c.set_markersize(3.6)
            c.set_markeredgewidth(1.0)

    ax.set_xscale("log")
    ax.set_yscale("log")

    xc = None
    if mark_crossover:
        xm, rm = paired(data, "matsim")
        if xm.size:
            xc = crossover(xm, rm)
    if xc:
        lo, hi, mid = xc
        ax.axvline(mid, color=pal["muted"], lw=0.8, ls=(0, (2, 2)), zorder=1)
        if annotate_crossover:
            # Both labels sit on one baseline in the empty strip above the x
            # axis, flanking the line. Naming the losing side is the point: the
            # claim is a regime with a crossover, not a constant speed-up.
            if regime_labels:
                ax.annotate("MATSim faster", xy=(mid, 0.015),
                            xycoords=("data", "axes fraction"),
                            xytext=(-4, 0), textcoords="offset points",
                            fontsize=6.0, color=pal["muted"], ha="right",
                            va="bottom", linespacing=1.15, zorder=7)
            ax.annotate(
                ("TorchDNL faster"
                 if regime_labels else f"crossover\n{mid / 1e3:.0f}k travellers"),
                xy=(mid, 0.015), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points",
                fontsize=6.0, color=pal["muted"], ha="left", va="bottom",
                linespacing=1.15, zorder=7,
            )

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=7.5, color=pal["muted"], pad=4)

    # Chrome: the house treatment -- two spines, light dashed major rules on
    # both axes, minor ticks killed outright.
    house_axes(ax, pal)

    # Decade ticks only, written as 10k / 1M rather than 10^4: the exponent
    # labels read as noise at column width and nobody converts them in their
    # head while comparing two panels.
    xs = np.concatenate([data[k][0] for k in data])
    ys = np.concatenate([data[k][1] for k in data])
    if censored:
        ys = np.concatenate([ys, np.array(list(censored.values()))])
    xt = list(xticks) if xticks is not None else _decade_ticks(xs.min(), xs.max())
    ax.set_xticks(xt)
    # Rotated like the reference: on a log axis the labels crowd together at
    # the right end, and 45 degrees keeps them apart without shrinking them.
    ax.set_xticklabels([xtick_fmt(t) for t in xt], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(_decade_ticks(ys.min(), ys.max()))
    ax.set_yticklabels([_fmt_seconds(t) for t in _decade_ticks(ys.min(), ys.max())])
    ax.set_xlim(xs.min() / 1.55, xs.max() * 1.55)
    ax.set_ylim(ys.min() / 1.7, ys.max() * 2.0)
    return xc


def legend_handles(pal, data_keys, usetex, censored=False):
    h = []
    for key, name in SERIES:
        if key not in data_keys:
            continue
        s = HOUSE_STROKE[key]
        label = LABEL_TEX[key] if (usetex and key in LABEL_TEX) else name
        h.append(mlines.Line2D([], [], color=pal[key], ls=s["ls"], lw=s["lw"],
                               marker=s["marker"], ms=s["ms"], mfc=pal[key],
                               mec=pal[key], mew=1.2, label=label))
    if censored:
        h.append(mlines.Line2D([], [], color=pal["sumo"], ls="none", marker="X",
                               ms=5.0, mfc="none", mec=pal["sumo"], mew=1.1,
                               label="SUMO did not finish"))
    return h


def finish(fig, out_stem, dark):
    path = f"{out_stem}.{'png' if dark else 'pdf'}"
    if dark:
        # Transparent so the deck's own ground shows through; the axes patch
        # must be cleared too or matplotlib paints it white under the data.
        fig.patch.set_alpha(0.0)
        for ax in fig.axes:
            ax.patch.set_alpha(0.0)
        fig.savefig(path, format="png", dpi=400, transparent=True,
                    bbox_inches="tight", pad_inches=0.03)
    else:
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_city(data, censored, title, out_stem, usetex):
    """One city, single column (3.3 in)."""
    for dark in (False, True):
        pal = house_palette(dark)
        fig, ax = plt.subplots(figsize=(3.3, 2.75))
        xc = draw_panel(ax, data, pal, censored=censored,
                        xlabel="Travellers Simulated",
                        ylabel="Computation Time (s)", title=title,
                        usetex=usetex)
        house_legend(ax, legend_handles(pal, data, usetex, bool(censored)), pal,
                     fontsize=6.2, loc="upper left", bbox_to_anchor=(-0.02, 1.03),
                     labelspacing=0.28, borderpad=0.0)
        fig.tight_layout(pad=0.35)
        finish(fig, out_stem, dark)
    return xc


def fig_cities(berlin, la, out_stem, usetex):
    r"""Berlin on top, Los Angeles below, shared y, one legend above both --
    stacked inside a SINGLE column (3.3 in), like `fig_grid`.

    Side by side at full text width gave each city 3.5 in and a `figure*` that
    had to float to the top of a page away from the table it supports. Stacked,
    the figure sits in the column next to its own paragraph. Two panels across
    3.3 in is not an option here either: the city titles alone are 30 characters.

    Only the bottom panel carries the x label -- both axes count travellers, so
    writing it twice spends a line of column height on a repeat. The tick labels
    stay on both: the two cities do NOT share an x range (Berlin stops at 3.5M,
    LA runs to 7.9M), so each panel's ticks are its own.
    """
    for dark in (False, True):
        pal = house_palette(dark)
        fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.6))
        for ax, (d, cen, ttl, xl) in zip(axes, [
            (berlin, SUMO_CENSORED["berlin"], "Berlin -- 159,039 car links",
             None),
            (la, SUMO_CENSORED["la"], "Los Angeles -- 1,154,727 car links",
             "Travellers Simulated"),
        ]):
            draw_panel(ax, d, pal, censored=cen, xlabel=xl,
                       ylabel="Computation Time (s)", title=ttl, usetex=usetex,
                       regime_labels=True)
        for ax in axes:
            ax.minorticks_off()
        _share_y(axes, pal)
        fig.align_ylabels(axes)
        # Legend above both panels. Two columns, not one row: four labels plus
        # the censoring key across 3.3 in would set at 4 pt to fit.
        fig.tight_layout(pad=0.35, h_pad=1.1, rect=(0, 0, 1, 0.925))
        house_legend(fig, legend_handles(pal, berlin, usetex, True), pal,
                     fontsize=7, ncol=2, loc="upper center",
                     bbox_to_anchor=(0.5, 1.012), handlelength=1.9,
                     handletextpad=0.45, columnspacing=1.4)
        finish(fig, out_stem, dark)


def fig_grid(pop, net, out_stem, usetex):
    r"""Synthetic grid: population sweep over a fixed network, then network-size
    sweep at fixed population.

    Stacked, not side by side: the paper puts this one inside a single column,
    and two panels across 3.3 in leaves each about 1.5 in wide, which collides
    the two titles and the 500k/1M tick labels. Drawing it wide and letting
    LaTeX shrink it to \columnwidth is what left the previous version with
    3 pt labels on the page.
    """
    for dark in (False, True):
        pal = house_palette(dark)
        fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.0))
        draw_panel(axes[0], pop, pal, mark_crossover=False,
                   xlabel="Population (Agents)", xticks=pop["torchdnl_gpu"][0],
                   ylabel="Computation Time (s)",
                   title=r"Demand sweep -- fixed 256$\times$256 grid" if usetex
                         else "Demand sweep -- fixed 256x256 grid", usetex=usetex)
        draw_panel(axes[1], net, pal, mark_crossover=False,
                   xlabel=r"Network Size (Grid $N{\times}N$)" if usetex
                          else "Network Size (Grid $NxN$)",
                   ylabel="Computation Time (s)",
                   title="Network sweep -- fixed 100,000 agents", usetex=usetex,
                   xtick_fmt=lambda v, _=None: f"{v:g}")
        # The network panel's x values are grid sides (32..1024), all powers of
        # two: decade ticks would put one label on the whole axis, so label the
        # measured points themselves.
        xs = net["torchdnl_gpu"][0]
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels([f"{int(v)}" for v in xs], rotation=45,
                                ha="right", rotation_mode="anchor")
        for ax in axes:
            ax.minorticks_off()
        _share_y(axes, pal)
        fig.align_ylabels(axes)
        # Legend above both panels, as in the reference. Two columns rather than
        # one row: four labels across 3.3 in would set at 4 pt to fit.
        fig.tight_layout(pad=0.35, h_pad=1.1, rect=(0, 0, 1, 0.915))
        house_legend(fig, legend_handles(pal, pop, usetex), pal, fontsize=7,
                     ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.012),
                     handlelength=1.9, handletextpad=0.45, columnspacing=1.4)
        finish(fig, out_stem, dark)


# ---------------------------------------------------------------------------
def report(name, data, xc):
    print(f"\n--- {name} ---")
    for key, label in SERIES:
        if key not in data:
            continue
        x, y, e, n = data[key]
        pts = ", ".join(f"{_fmt_count(a)}:{b:.0f}s" for a, b in zip(x, y))
        print(f"  {label:<20} n={n.min()}-{n.max()} seeds  {pts}")
    for key, label in SERIES:
        if key == "torchdnl_gpu":
            continue
        x, r = paired(data, key)
        if not x.size:
            continue
        c = crossover(x, r)
        cs = (f"bracket [{c[0]:,.0f}; {c[1]:,.0f}] -> {c[2]:,.0f}" if c
              else "no crossover in range")
        print(f"  {label:<20} max {r.max():.2f}x slower than GPU at "
              f"{x[int(r.argmax())]:,.0f}; crossover {cs}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--berlin-dir", required=True)
    p.add_argument("--la-dir", required=True)
    p.add_argument("--grid-dir", default="graphs/data")
    p.add_argument("--out-dir", default="aamas_2027_template/Figures")
    a = p.parse_args()

    usetex = setup_style()
    os.makedirs(a.out_dir, exist_ok=True)
    o = lambda n: os.path.join(a.out_dir, n)  # noqa: E731

    berlin = load_city(a.berlin_dir)
    la = load_city(a.la_dir)
    pop = load_grid(f"{a.grid_dir}/scale_population_benchmark.csv",
                    "population", lambda s: int(s))
    # 3x3 and 16x16 are dropped: 16x16 has no MATSim cell, and 3x3's CPU-DNL
    # time is a pre-fix outlier (385 s, above the 1024x1024 figure's neighbours).
    net = load_grid(f"{a.grid_dir}/scale_network_benchmark.csv",
                    "size", lambda s: int(s.split("x")[0]))
    net = {k: tuple(arr[v[0] >= 32] for arr in v) for k, v in net.items()}

    print("rendering:")
    xb = fig_city(berlin, SUMO_CENSORED["berlin"],
                  "Berlin -- 159,039 car links", o("scaling_law"), usetex)
    xl = fig_city(la, SUMO_CENSORED["la"],
                  "Los Angeles -- 1,154,727 car links", o("scaling_law_la"), usetex)
    fig_cities(berlin, la, o("scaling_law_cities"), usetex)
    fig_grid(pop, net, o("combined_scalability"), usetex)

    report("Berlin", berlin, xb)
    report("Los Angeles", la, xl)
    report("Grid, population sweep", pop, None)
    report("Grid, network sweep", net, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
