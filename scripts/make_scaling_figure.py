"""Build the wall-clock scaling figure: TorchDNL vs MATSim across a population
ladder on a real network.

The claim the figure carries is that the GPU advantage is a REGIME, not a
constant. TorchDNL advances a fixed number of timesteps whatever the demand,
while MATSim's event-driven queue pays per link traversal, so the two curves have
different shapes and cross at a measurable population.

Every figure is emitted twice, always:

  <output>.pdf   light theme, for the article
  <output>.png   TRANSPARENT background, light ink, for dark-background slides

Data-series colours are identical in both so the figure stays recognisable from
paper to deck; only the ink, gridlines and background change.

Usage:
    python scripts/make_scaling_figure.py \\
        --bench-dir results/bench_berlin36 \\
        --network-label "Berlin, 202,939 links" \\
        --output Figures/scaling_law
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from tamarl.visualisation.validation_figures import (
    COLOR_AQUA,
    COLOR_MATSIM,
    COLOR_TORCHDNL,
    setup_style,
)

# The four series, in legend order. `key` is the engine label this module uses
# internally; see `load()` for why it is derived from the tag and not from the
# CSV's own `engine` column.
SERIES = (
    ("matsim", "MATSim", "-"),
    ("sumo", "SUMO (meso)", "-"),
    ("torchdnl_gpu", r"\texttt{TorchDNL} (1$\times$H100)", "-"),
    ("torchdnl_cpu", r"\texttt{TorchDNL} (CPU)", (0, (4, 1.5))),
)

# Light ink for dark slides. The data colours are lightened just enough to hold
# contrast against a dark ground without losing their hue identity.
DARK_INK = "#ebe9e4"
DARK_MUTED = "#a9a69f"
DARK_GRID = "#3a3a3a"
DARK_MATSIM = "#5b9df0"
DARK_TORCHDNL = "#ff8752"
DARK_SUMO = "#3ddba0"


def _series_key(row: dict) -> str:
    """Which curve a row belongs to.

    The CSV's `engine` column says `torchdnl` for BOTH the GPU and the CPU run --
    they are the same engine invoked with a different `--device`, so the runner
    records the same name. Only the cell tag carries the distinction, and merging
    the two would average a 15x gap into nonsense.
    """
    engine = row["engine"]
    if engine == "torchdnl":
        return "torchdnl_cpu" if "_torchdnl_cpu_" in row["tag"] else "torchdnl_gpu"
    return engine


def load(bench_dir: str) -> dict:
    """Return {series: (travellers[], mean_engine_s[], std[])} sorted by population."""
    rows = []
    for f in glob.glob(f"{bench_dir}/*_bench.csv"):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        raise SystemExit(f"ABORT: no *_bench.csv under {bench_dir}")

    # Traveller counts only appear on the TorchDNL rows (neither MATSim's nor
    # SUMO's runner reports them), so map loading -> travellers from those and
    # reuse the mapping for every series. All four engines are handed the same
    # population file at a given loading, so the count is shared by construction.
    n_by_load = {
        float(r["loading"]): int(float(r["n_agents"]))
        for r in rows
        if r["engine"] == "torchdnl" and r.get("n_agents")
    }

    grouped = defaultdict(list)
    for r in rows:
        t = r.get("t_engine_s")
        if not t or str(t).lower() == "nan":
            continue
        grouped[(_series_key(r), float(r["loading"]))].append(float(t))

    out = {}
    for key, _, _ in SERIES:
        loads = sorted(L for (e, L) in grouped if e == key and L in n_by_load)
        if not loads:
            continue
        out[key] = (
            np.array([n_by_load[L] for L in loads], dtype=float),
            np.array([st.mean(grouped[(key, L)]) for L in loads]),
            np.array([
                st.stdev(grouped[(key, L)]) if len(grouped[(key, L)]) > 1 else 0.0
                for L in loads
            ]),
        )
    return out


def crossover(x, ratio) -> float | None:
    """Population at which the ratio first reaches 1, by log-linear interpolation."""
    for i in range(1, len(ratio)):
        if ratio[i - 1] < 1.0 <= ratio[i]:
            lo, hi = np.log10(x[i - 1]), np.log10(x[i])
            f = (1.0 - ratio[i - 1]) / (ratio[i] - ratio[i - 1])
            return float(10 ** (lo + f * (hi - lo)))
    return None


def paired(data: dict, key: str) -> tuple:
    """Align one series against TorchDNL-GPU on the loadings BOTH actually have.

    Series are stored per engine, each sorted by its own loading set, so dividing
    them elementwise by position silently compares different populations whenever
    the sets differ -- which happens the moment one engine's cell for a loading has
    landed and another's has not. That produced a reported 9.43x "at 1,736,000
    travellers" against a measured 5.21x there, by pairing a MATSim point with a
    TorchDNL point from a different loading. Matching on the loading VALUE is the
    only safe way.

    Returns (travellers, ratio) over the shared loadings, ascending.
    """
    if key not in data or "torchdnl_gpu" not in data:
        return np.array([]), np.array([])
    xo, yo, _ = data[key]
    xt, yt, _ = data["torchdnl_gpu"]
    # Traveller count is the population identity here: the same loading always maps
    # to the same count, and it is what the x axis shows.
    ref = dict(zip(xt, yt))
    shared = [(x, y / ref[x]) for x, y in zip(xo, yo) if x in ref]
    if not shared:
        return np.array([]), np.array([])
    shared.sort()
    return (np.array([x for x, _ in shared]), np.array([r for _, r in shared]))


def render(data: dict, label: str, out_path: str, dark: bool) -> None:
    ink = DARK_INK if dark else "#0b0b0b"
    muted = DARK_MUTED if dark else "#52514e"
    grid = DARK_GRID if dark else "#e1e0d9"
    colors = {
        "matsim": DARK_MATSIM if dark else COLOR_MATSIM,
        "sumo": DARK_SUMO if dark else COLOR_AQUA,
        "torchdnl_gpu": DARK_TORCHDNL if dark else COLOR_TORCHDNL,
        "torchdnl_cpu": DARK_TORCHDNL if dark else COLOR_TORCHDNL,
    }

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for key, name, ls in SERIES:
        if key not in data:
            continue
        x, y, e = data[key]
        ax.plot(x, y, marker="o", ms=3.5, lw=1.6, ls=ls, color=colors[key],
                label=name, zorder=3)
        ax.fill_between(x, y - e, y + e, color=colors[key], alpha=0.20, lw=0, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    xc = None
    xm_p, r_p = paired(data, "matsim")
    if xm_p.size:
        xc = crossover(xm_p, r_p)
    if xc:
        ax.axvline(xc, color=muted, lw=0.8, ls=":", zorder=1)
        # Anchor y in AXES fraction, not data: on a log axis the data limits are
        # not settled until the scale is set, and an absolute y placed before that
        # lands off-canvas and is silently clipped.
        ax.annotate(
            f"crossover\n{xc / 1e3:.0f}k travellers",
            xy=(xc, 0.02), xycoords=("data", "axes fraction"),
            xytext=(5, 2), textcoords="offset points",
            fontsize=6.5, color=muted, ha="left", va="bottom",
        )
    ax.set_xlabel("Travellers simulated")
    ax.set_ylabel("Engine wall-clock (s)")
    ax.set_title(label, fontsize=8, color=muted, pad=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink, which="both")
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    ax.grid(axis="y", color=grid, linewidth=0.7, zorder=0)
    leg = ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(ink)

    fig.tight_layout(pad=0.4)
    if dark:
        # Transparent so the deck's own background shows through; the axes patch
        # must be cleared too, or matplotlib paints it white under the data.
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        fig.savefig(out_path, format="png", dpi=300, transparent=True,
                    bbox_inches="tight", pad_inches=0.03)
    else:
        fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--network-label", default="Berlin")
    p.add_argument("--output", required=True, help="Path stem; .pdf and .png are appended")
    args = p.parse_args()

    data = load(args.bench_dir)
    setup_style()
    stem = args.output[:-4] if args.output.endswith((".pdf", ".png")) else args.output
    render(data, args.network_label, stem + ".pdf", dark=False)
    render(data, args.network_label, stem + ".png", dark=True)

    if "torchdnl_gpu" not in data:
        return 0
    for key, name, _ in SERIES:
        if key == "torchdnl_gpu":
            continue
        x, r = paired(data, key)
        if not x.size:
            continue
        xc = crossover(x, r)
        print(f"{name}: {len(x)} shared loadings; max speed-up {r.max():.2f}x at "
              f"{x[int(r.argmax())]:,.0f} travellers; crossover at "
              f"{f'{xc:,.0f}' if xc else 'never/always'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
