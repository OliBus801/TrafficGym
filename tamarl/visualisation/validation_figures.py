"""Publication-quality (paper supplementary material) figures for the
TorchDNL-vs-MATSim DNL-fidelity validation appendix.

This module is presentation-only: every number plotted here is produced by
`tamarl.validation.geh` / `tamarl.validation.distributional` / the parsers in
`tamarl.validation.matsim_io`, exactly as in `scripts/validate_against_matsim.py`
(which writes the companion `summary.md`). Nothing here recomputes metrics —
it only lays them out.

Design notes (see the repo's dataviz skill for the full method):
  - Two-color categorical palette, held fixed across every figure:
    MATSim = slot-1 blue (#2a78d6), TorchDNL = slot-2 orange (#eb6834).
    Validated colorblind-safe (adjacent CVD dE 24.7, normal-vision dE 33.6,
    both well above the dE>=8 / dE>=15 gates).
  - The four-series GEH figure reuses categorical slots 1-4 (blue, orange,
    aqua, yellow); slots 3-4 sit under the 3:1 contrast floor on a light
    surface, so that figure ships direct end-labels + a legend (the required
    "relief channel") rather than color alone.
  - Vector PDF output only (no raster), serif/mathtext publication font,
    sized for direct \\includegraphics at typical single-column widths.
  - Captions live in the paper's LaTeX source, so figures carry precise axis
    labels/legends but no in-figure titles (consistent across the set).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Palette (see tamarl/data/scenarios/sioux_falls/dnl_validation/report/ for
# the validator output backing these choices).
# --------------------------------------------------------------------------
COLOR_MATSIM = "#2a78d6"      # categorical slot 1 (blue)
COLOR_TORCHDNL = "#eb6834"    # categorical slot 2 (orange)
COLOR_AQUA = "#1baf7a"        # categorical slot 3 - used only in the 4-series GEH figure
COLOR_YELLOW = "#eda100"      # categorical slot 4 - used only in the 4-series GEH figure

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Dark-slide counterparts (transparent PNG output). Same hues, lifted just
# enough to hold contrast on a dark ground — see scripts/make_scaling_figure.py,
# which established this dual-output convention.
DARK_INK = "#ebe9e4"
DARK_SUB = "#bcb9b2"
DARK_MUTED = "#a9a69f"
DARK_GRID = "#343434"
DARK_BASE = "#5c5a55"
DARK_MATSIM = "#5b9df0"
DARK_TORCHDNL = "#ff8752"

GEH_SERIES_COLORS = {
    "en_route": COLOR_MATSIM,
    "departures": COLOR_TORCHDNL,
    "arrivals": COLOR_AQUA,
    "link_volume": COLOR_YELLOW,
}
GEH_SERIES_LABELS = {
    "en_route": "En-route agents",
    "departures": "Departures",
    "arrivals": "Arrivals",
    "link_volume": "Link volume (full-day)",
}
GEH_SERIES_LINESTYLES = {
    "en_route": "-",
    "departures": "--",
    "arrivals": "-.",
    "link_volume": ":",
}

_STYLE_APPLIED = False

# --------------------------------------------------------------------------
# HOUSE STYLE -- the authoritative look for every figure in the paper.
#
# Lifted verbatim from `graphs/notebooks/performance_benchmark_graphs.ipynb`,
# cell 0 (light) and cell 2 (dark slides). That cell drew
# the paper repository's `Figures/combined_scalability.pdf`, which the
# authors picked as the reference: it is the one figure whose look is settled,
# so everything else is drawn from these tokens rather than from its own
# bespoke palette.
#
# The contract: one engine -> one colour, one marker and one dash pattern,
# everywhere. The two TorchDNL series share a blue hue family because they are
# the same code on two devices; marker and dash separate them.
#
# Deliberate deviation from the notebook: FONT SIZES are not copied. The
# notebook draws at 6.5 in with 9-10 pt type and LaTeX then shrinks the figure
# to 0.78\columnwidth (x0.40), so its labels land on the page at ~4 pt. Every
# figure here is drawn at its FINAL printed size instead, so type is set to
# land at the size it is actually read at (see `setup_style`). Colours,
# markers, dash patterns, line widths, marker sizes, grid and legend treatment
# are the notebook's exact values.
# --------------------------------------------------------------------------
HOUSE_LIGHT = {
    "torchdnl_gpu": "#1f77b4",   # tab:blue        -- "TorchDNL (ours)"
    "torchdnl_cpu": "#aec7e8",   # tab:blue light  -- "TorchDNL (CPU)"
    "matsim": "#ff7f0e",         # tab:orange
    "sumo": "#d62728",           # tab:red
    "ink": "#000000",
    "sub": "#52514e",
    "muted": "#595959",
    "faint": "#8a8a8a",
    "base": "#bdbdbd",           # y=x / baseline rules
    "grid": "#DDDDDD",           # notebook: grid.color
    "band": "#bfbfbf",
    "band_alpha": 0.22,
    "box": "#ffffff",
}

# Dark-slide counterparts, from the notebook's own dark cell: same series
# identity, hues lifted just enough to hold contrast on a dark ground.
HOUSE_DARK = {
    "torchdnl_gpu": "#58a6ff",
    "torchdnl_cpu": "#aec7e8",
    "matsim": "#ff9f43",
    "sumo": "#ff6b6b",
    "ink": "#ffffff",
    "sub": "#d8d6d1",
    "muted": "#bcb9b2",
    "faint": "#CCCCCC",          # notebook: axes.edgecolor on dark
    "base": "#777777",
    "grid": "#444444",           # notebook: grid.color on dark
    "band": "#9a9a9a",
    "band_alpha": 0.16,
    "box": "#1a1a1a",
}

# Per-engine stroke identity: (linestyle, marker, markersize, linewidth,
# zorder). Exactly the notebook's `styles` dict.
HOUSE_STROKE = {
    "torchdnl_gpu": {"ls": "-", "marker": "o", "ms": 6, "lw": 2.0, "z": 5},
    "torchdnl_cpu": {"ls": "--", "marker": "s", "ms": 5, "lw": 1.2, "z": 4},
    "matsim": {"ls": ":", "marker": "^", "ms": 5, "lw": 1.2, "z": 3},
    "sumo": {"ls": "-.", "marker": "x", "ms": 5, "lw": 1.2, "z": 2},
}

# Grid: light dotted-dashed rules on BOTH axes, major ticks only, minor ticks
# suppressed outright (the notebook calls this "suppression radicale").
HOUSE_GRID = {"linestyle": "--", "linewidth": 0.5}


def house_palette(dark: bool) -> dict:
    """The house colour tokens for the light PDF or the dark-slide PNG."""
    return dict(HOUSE_DARK if dark else HOUSE_LIGHT)


def house_axes(ax, pal: dict, *, x_grid: bool = True, y_grid: bool = True,
               minor_ticks: bool = False) -> None:
    """Apply the reference figure's chrome to one panel.

    Two spines, ink-coloured labels on faint ticks, light dashed major grid on
    both axes, no minor-tick noise, data over the grid.
    """
    import matplotlib.ticker as mticker

    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(pal["faint"])
    ax.tick_params(colors=pal["faint"], labelcolor=pal["ink"], which="major",
                   length=2.5, pad=2)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.label.set_color(pal["ink"])
    ax.yaxis.label.set_color(pal["ink"])
    if y_grid:
        ax.grid(True, axis="y", which="major", color=pal["grid"],
                linestyle=HOUSE_GRID["linestyle"],
                linewidth=HOUSE_GRID["linewidth"], zorder=0)
    if x_grid:
        ax.grid(True, axis="x", which="major", color=pal["grid"],
                linestyle=HOUSE_GRID["linestyle"],
                linewidth=HOUSE_GRID["linewidth"], zorder=0)
    if not minor_ticks:
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_axisbelow(True)


def house_legend(fig_or_ax, handles, pal: dict, **kw):
    """A single frameless legend row, labels in ink. The reference parks it
    above the axes; callers pass their own loc/bbox/ncol."""
    opts = dict(frameon=False, handlelength=2.0, handletextpad=0.5,
                columnspacing=1.6, labelspacing=0.3, borderaxespad=0.0)
    opts.update(kw)
    leg = fig_or_ax.legend(handles=handles, **opts)
    for t in leg.get_texts():
        t.set_color(pal["ink"])
    return leg


def setup_style() -> bool:
    """Configure matplotlib rcParams for camera-ready vector figures.

    Tries LaTeX (`text.usetex`) first for true publication typesetting;
    falls back to matplotlib's built-in STIX math/serif fonts (no external
    LaTeX install required) if a quick smoke-test render fails.

    Returns:
        True if `text.usetex` is active, False if the STIX fallback is used
        (callers may want to report this to the user as a caveat).
    """
    global _STYLE_APPLIED
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,  # embed as proper (non-Type3) outlines
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
            # House grid + legend, from the reference notebook's rcParams.
            "grid.color": HOUSE_LIGHT["grid"],
            "grid.linestyle": HOUSE_GRID["linestyle"],
            "grid.linewidth": HOUSE_GRID["linewidth"],
            "legend.frameon": False,
            "axes.axisbelow": True,
        }
    )

    usetex_ok = _try_usetex()
    if usetex_ok:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif"]

    _STYLE_APPLIED = True
    return usetex_ok


def _try_usetex() -> bool:
    """Smoke-test that a full LaTeX toolchain (latex + dvipng/type1cm etc.)
    actually works with matplotlib, rather than assuming from `which latex`."""
    import io

    import matplotlib

    saved = dict(matplotlib.rcParams)
    try:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["font.family"] = "serif"
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel(r"Test $x^2$ label")
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf")
        plt.close(fig)
        return True
    except Exception:
        return False
    finally:
        matplotlib.rcParams.update(saved)


def _clean_axes(ax, x_grid=False, y_grid=True):
    """Apply the shared spine/gridline/tick treatment to every figure."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_PRIMARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    if y_grid:
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
    if x_grid:
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, output_path: str) -> None:
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 1: en-route envelope
# --------------------------------------------------------------------------
def plot_en_route_envelope(
    bins: np.ndarray,
    matsim_en_route: np.ndarray,  # [num_seeds, num_bins]
    torchdnl_en_route: np.ndarray,  # [num_seeds, num_bins]
    output_path: str,
) -> None:
    """Symmetric two-envelope comparison: MATSim's inter-seed IQR band vs
    TorchDNL's own inter-seed IQR band, both over hour-of-day, plus each
    simulator's median. Avoids the clutter of 10 overlapping per-seed lines
    while still showing each simulator's own stochastic spread."""
    import matplotlib.pyplot as plt

    from tamarl.validation import distributional

    m_median, m_lower, m_upper = distributional.build_envelope(matsim_en_route)
    t_median, t_lower, t_upper = distributional.build_envelope(torchdnl_en_route)

    x = bins[:-1] / 3600.0

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    _clean_axes(ax)

    ax.fill_between(
        x, m_lower, m_upper, step="post", color=COLOR_MATSIM, alpha=0.18,
        linewidth=0, zorder=1, label="MATSim (10-seed IQR)",
    )
    ax.fill_between(
        x, t_lower, t_upper, step="post", color=COLOR_TORCHDNL, alpha=0.18,
        linewidth=0, zorder=1, label="TorchDNL (10-seed IQR)",
    )
    ax.step(x, m_median, where="post", color=COLOR_MATSIM, linewidth=1.4, zorder=3, label="MATSim median")
    ax.step(x, t_median, where="post", color=COLOR_TORCHDNL, linewidth=1.4, zorder=3, label="TorchDNL median",
             linestyle=(0, (4, 1.5)))

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Agents en route")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    if x.max() - x.min() >= 12:
        ax.xaxis.set_major_locator(plt.MultipleLocator(4))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    # Both AM and PM peaks reach close to the axis ceiling, so the legend is
    # anchored over the (empty) inter-peak trough rather than a corner.
    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.47, 1.0), ncol=2, columnspacing=1.0,
        frameon=True, facecolor="white", edgecolor=GRIDLINE, framealpha=0.9,
        handlelength=1.5, labelspacing=0.3, borderaxespad=0.2,
    )
    legend.get_frame().set_linewidth(0.7)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 2: pooled trip travel-time ECDF
# --------------------------------------------------------------------------
def plot_travel_time_ecdf(
    matsim_tt: np.ndarray,
    torchdnl_tt: np.ndarray,
    wasserstein: float,
    ks_stat: float,
    ks_pvalue: float,
    output_path: str,
) -> None:
    """ECDF comparison of pooled trip travel times. An ECDF reads more
    rigorously than overlapping histograms here: it needs no bin-width
    choice, makes the KS statistic (the max vertical gap between the two
    curves) directly visible, and two near-coincident step curves are an
    unambiguous "these are the same distribution" signal in a way that two
    translucent, bin-dependent histograms are not."""
    import matplotlib.pyplot as plt

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sorted = np.sort(values)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    m_x, m_y = ecdf(matsim_tt)
    t_x, t_y = ecdf(torchdnl_tt)

    # x-axis in minutes reads more naturally than raw seconds for trip times.
    m_x_min = m_x / 60.0
    t_x_min = t_x / 60.0
    x_cap = np.percentile(np.concatenate([m_x_min, t_x_min]), 99.5)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    _clean_axes(ax)

    ax.plot(m_x_min, m_y, color=COLOR_MATSIM, linewidth=1.5, drawstyle="steps-post",
            zorder=3, label="MATSim")
    ax.plot(t_x_min, t_y, color=COLOR_TORCHDNL, linewidth=1.5, drawstyle="steps-post",
            zorder=3, label="TorchDNL", linestyle=(0, (4, 1.5)))

    ax.set_xlabel("Trip travel time (min)")
    ax.set_ylabel("Cumulative fraction of trips")
    ax.set_xlim(0, x_cap)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    legend = ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    stat_text = (
        f"Wasserstein $=$ {wasserstein:.1f}\\,s\n"
        f"KS $=$ {ks_stat:.3f} ($p={ks_pvalue:.2g}$)"
    )
    ax.text(
        0.03, 0.97, stat_text, transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color=INK_SECONDARY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=GRIDLINE, linewidth=0.7),
    )

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 3: per-link full-day volume parity
# --------------------------------------------------------------------------
def plot_link_volume_parity(
    matsim_link_vol: np.ndarray,  # [num_seeds, num_links]
    torchdnl_link_vol: np.ndarray,  # [num_seeds, num_links]
    output_path: str,
) -> None:
    """Per-link full-day volume: TorchDNL vs MATSim, seed-averaged, one
    point per network link, against the y=x parity line. The strongest,
    most visually "irrefutable" result in the validation set (link-volume
    GEH mean/median ~= 0), so it gets its own dedicated scatter."""
    import matplotlib.pyplot as plt

    m_mean = matsim_link_vol.mean(axis=0)
    t_mean = torchdnl_link_vol.mean(axis=0)

    lim_max = max(m_mean.max(), t_mean.max()) * 1.05

    r = np.corrcoef(m_mean, t_mean)[0, 1]

    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    _clean_axes(ax, x_grid=True, y_grid=True)

    ax.plot([0, lim_max], [0, lim_max], color=BASELINE, linewidth=1.0, linestyle="-", zorder=1,
            label="$y = x$")
    ax.scatter(
        m_mean, t_mean, s=11, facecolor=COLOR_TORCHDNL, edgecolor="white", linewidth=0.3,
        alpha=0.75, zorder=3,
    )

    ax.set_xlabel("MATSim link volume (veh/day)")
    ax.set_ylabel("TorchDNL link volume (veh/day)")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.04, 0.96, f"$r = {r:.4f}$\n$n = {len(m_mean)}$ links",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK_SECONDARY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=GRIDLINE, linewidth=0.7),
    )
    legend = ax.legend(loc="lower right", frameon=False, handlelength=1.4, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 4: GEH-distribution summary across series
# --------------------------------------------------------------------------
def plot_geh_distribution(
    geh_values_by_series: dict[str, np.ndarray],
    output_path: str,
    order: tuple[str, ...] = ("en_route", "departures", "arrivals", "link_volume"),
) -> None:
    """ECDF of GEH values for each of the four validation series, with the
    standard traffic-engineering acceptance thresholds (GEH<5 good,
    GEH<10 marginal) marked as reference lines. Ships direct end-labels in
    addition to the legend since two of the four series colors sit under
    the 3:1 contrast floor on a light surface (see module docstring)."""
    import matplotlib.pyplot as plt

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sorted = np.sort(values)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    _clean_axes(ax)

    x_max = 0.0
    for name in order:
        values = np.asarray(geh_values_by_series[name])
        x, y = ecdf(values)
        x_max = max(x_max, np.percentile(x, 99.5))
        ax.plot(
            x, y, color=GEH_SERIES_COLORS[name], linestyle=GEH_SERIES_LINESTYLES[name],
            linewidth=1.4, drawstyle="steps-post", zorder=3, label=GEH_SERIES_LABELS[name],
        )

    x_max = max(x_max, 11.0)
    ax.axvline(5.0, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=2)
    ax.axvline(10.0, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=2)
    # Threshold labels sit near the top, clear of the lower-right legend box.
    ax.text(5.0, 0.93, "GEH$<$5", rotation=90, ha="right", va="top", fontsize=6, color=INK_MUTED)
    ax.text(10.0, 0.93, "GEH$<$10", rotation=90, ha="right", va="top", fontsize=6, color=INK_MUTED)

    ax.set_xlabel("GEH statistic")
    ax.set_ylabel("Cumulative fraction of observations")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)

    legend = ax.legend(
        loc="lower right", frameon=True, facecolor="white", edgecolor=GRIDLINE,
        framealpha=0.9, handlelength=1.8, labelspacing=0.3, fontsize=6.5,
        borderaxespad=0.3,
    )
    legend.get_frame().set_linewidth(0.7)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 5: cross-scenario summary grid (3 fidelity signals x N scenarios)
# --------------------------------------------------------------------------
def _theme(dark: bool) -> dict:
    """Ink/grid/data colours for the two mandated outputs: a light PDF for the
    paper and a transparent, light-ink PNG for dark slides.

    Straight off the HOUSE palette, so the two engines carry exactly the hues
    they carry in the scalability figures: TorchDNL is the reference's
    "TorchDNL (ours)" blue, MATSim the reference's orange. This figure compares
    two engines instead of four, but it speaks the same vocabulary.
    """
    pal = house_palette(dark)
    th = dict(pal)
    th["matsim"] = pal["matsim"]
    th["torchdnl"] = pal["torchdnl_gpu"]
    return th


def _style_panel(ax, th: dict, x_grid: bool = False, y_grid: bool = True) -> None:
    house_axes(ax, th, x_grid=x_grid, y_grid=y_grid, minor_ticks=True)


def _si(v: float) -> str:
    """Compact count tick label: 1200 -> 1.2k, 25000 -> 25k, 1.2e6 -> 1.2M."""
    a = abs(v)
    if a >= 1e6:
        s = f"{v / 1e6:.1f}M"
    elif a >= 1e3:
        s = f"{v / 1e3:.0f}k" if a >= 1e4 else f"{v / 1e3:.1f}k"
    else:
        s = f"{v:.0f}"
    return s.replace(".0k", "k").replace(".0M", "M")


def _stat_box(ax, lines: list[str], th: dict, loc: str = "upper left",
              fontsize: float = 6.0) -> None:
    """Small annotation block. No frame: on three near-perfect panels a boxed
    annotation reads as chartjunk, and the panels have empty corners anyway."""
    xy = {"upper left": (0.035, 0.97, "left", "top"),
          "lower right": (0.965, 0.04, "right", "bottom"),
          "upper right": (0.965, 0.97, "right", "top")}[loc]
    ax.text(xy[0], xy[1], "\n".join(lines), transform=ax.transAxes,
            ha=xy[2], va=xy[3], fontsize=fontsize, color=th["sub"], linespacing=1.35,
            zorder=6)


# --------------------------------------------------------------------------
# Figure 5: cross-scenario fidelity grid (3 signals x N scenarios)
# --------------------------------------------------------------------------
def plot_combined_scenario_grid(scenarios: list[dict], output_path: str,
                                dark: bool = False) -> None:
    """One cross-scenario DNL-fidelity figure: three fidelity signals (rows)
    against scenarios ordered left-to-right by increasing network size.

    Rows:
      0. Network-wide en-route count over the day, both engines (seed median).
         This is the "both engines trace the same day" visual. The MATSim
         inter-seed envelope is deliberately NOT drawn: with fixed routes and
         no re-planning MATSim is near-deterministic, so the band is ~0.1-0.2%
         wide and its inclusion rate says nothing about agreement (see
         RAPPORT_AAMAS section 3). Max GEH on the series is annotated instead,
         which is the metric the paper actually reports.
      1. Full-day per-link volume parity, seed-averaged, one point per link,
         on log-log axes because link volumes span four orders of magnitude
         and a linear parity plot collapses every scenario into a corner.
         Points are rasterised (LA alone contributes 1.15M of them).
      2. Pooled trip travel-time ECDFs with the Wasserstein distance. Only
         COMPLETED trips are compared on both sides: MATSim's trip file lists
         finished trips only, while TorchDNL imputes a time for agents still
         in the network at the horizon, so a raw comparison would contrast two
         different populations.

    Each element of `scenarios` is a dict with keys: `label`, `n_links`,
    `n_seeds`, `bins`, `matsim_en_route`, `torchdnl_en_route`,
    `matsim_link_vol`, `torchdnl_link_vol`, `matsim_trip_tt`,
    `torchdnl_trip_tt`.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    from tamarl.validation import distributional
    from tamarl.validation import geh as geh_mod

    th = _theme(dark)
    n_cols = len(scenarios)
    fig, axes = plt.subplots(3, n_cols, figsize=(6.3, 5.55))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sorted = np.sort(values)
        return x_sorted, np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    for col, sc in enumerate(scenarios):
        # ---- Row 0: en-route time series ----
        ax = axes[0, col]
        _style_panel(ax, th)
        m_med = np.median(sc["matsim_en_route"], axis=0)
        t_med = np.median(sc["torchdnl_en_route"], axis=0)
        x = sc["bins"][:-1] / 3600.0

        # The caches run to a 36 h horizon so the demand can drain, but the
        # empty tail is half the axis on every scenario. Clip to the loaded
        # span (plus one bin of air) or the peaks shrink to nothing.
        loaded = np.flatnonzero((m_med > 0) | (t_med > 0))
        lo, hi = (x[loaded[0]], x[min(loaded[-1] + 2, len(x) - 1)]) if loaded.size else (x[0], x[-1])
        lo = max(0.0, lo - 0.5)

        # House stroke vocabulary: TorchDNL is the thick solid line (the
        # reference's "ours"), MATSim the thin dotted one laid over it, so the
        # agreement reads as one curve tracing the other.
        ax.step(x, t_med, where="post", color=th["torchdnl"], zorder=3,
                linewidth=HOUSE_STROKE["torchdnl_gpu"]["lw"],
                linestyle=HOUSE_STROKE["torchdnl_gpu"]["ls"],
                solid_joinstyle="round")
        ax.step(x, m_med, where="post", color=th["matsim"], zorder=4,
                linewidth=HOUSE_STROKE["matsim"]["lw"],
                linestyle=HOUSE_STROKE["matsim"]["ls"])
        ax.set_xlim(lo, hi)
        # Headroom for the corner annotation: without it the stat block sits on
        # top of the peak, which is the one part of the curve that must be clear.
        peak = max(m_med.max(), t_med.max())
        ax.set_ylim(0, peak * 1.46)
        span = hi - lo
        ax.xaxis.set_major_locator(
            plt.MultipleLocator(6 if span >= 18 else 4 if span >= 10 else 2)
            if span >= 4 else plt.MaxNLocator(nbins=4)
        )
        # Tick the DATA range, not the padded range: letting the locator see the
        # 1.46x ceiling adds a gridline above every curve and strands the
        # annotation on it.
        yticks = plt.MaxNLocator(nbins=4).tick_values(0, peak)
        ax.set_yticks([t for t in yticks if 0 <= t <= peak * 1.02])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _si(v)))

        g_en = geh_mod.geh_series_from_counts(sc["torchdnl_en_route"], sc["matsim_en_route"])
        pct = geh_mod.acceptance_rate(g_en) * 100.0
        pct_s = "100" if pct >= 99.95 else f"{pct:.1f}"
        _stat_box(ax, [f"peak {_si(peak)} en route",
                       rf"GEH$<$5: {pct_s}\% of bins",
                       f"max GEH {g_en.max():.2f}"], th, "upper right")

        ax.set_title(sc["label"], fontsize=9, color=th["ink"], pad=12, fontweight="bold")
        ax.annotate(
            f"{sc['n_links']:,} links  ·  {sc['n_seeds']} matched seeds",
            xy=(0.5, 1.015), xycoords="axes fraction", ha="center", va="bottom",
            fontsize=6.5, color=th["sub"],
        )
        ax.set_xlabel("Hour of day", fontsize=7)
        if col == 0:
            ax.set_ylabel("Agents en route", fontsize=7.5)

        # ---- Row 1: per-link full-day volume parity ----
        ax = axes[1, col]
        _style_panel(ax, th, x_grid=True)
        m_mean = sc["matsim_link_vol"].mean(axis=0)
        t_mean = sc["torchdnl_link_vol"].mean(axis=0)
        r = np.corrcoef(m_mean, t_mean)[0, 1]

        # Links idle in BOTH engines carry no information and would stack on
        # the axis origin; links idle in only one DO matter, so they are
        # floored onto the axis edge rather than dropped.
        used = (m_mean > 0) | (t_mean > 0)
        floor = 0.5
        mx = np.maximum(m_mean[used], floor)
        ty = np.maximum(t_mean[used], floor)
        top = max(mx.max(), ty.max()) * 1.9
        # Data-driven lower bound: a fixed decade floor leaves the whole
        # lower-left of the Sioux Falls panel empty (its lightest link still
        # carries hundreds of vehicles).
        bot = min(mx.min(), ty.min()) * 0.55

        ax.plot([bot, top], [bot, top], color=th["base"], linewidth=0.9, zorder=1)
        ax.scatter(mx, ty, s=2.2, color=th["torchdnl"], linewidth=0,
                   alpha=0.10 if used.sum() > 2e5 else 0.35 if used.sum() > 2e4 else 0.75,
                   zorder=3, rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(bot, top)
        ax.set_ylim(bot, top)
        ax.set_aspect("equal", adjustable="box")
        _stat_box(ax, [f"$r = {r:.5f}$", f"{used.sum():,} loaded links"], th, "upper left")
        ax.set_xlabel("MATSim volume (veh/day)", fontsize=7)
        if col == 0:
            ax.set_ylabel("TorchDNL volume\n(veh/day)", fontsize=7.5)

        # ---- Row 2: pooled completed-trip travel-time ECDF ----
        ax = axes[2, col]
        _style_panel(ax, th)
        m_tt, t_tt = sc["matsim_trip_tt"], sc["torchdnl_trip_tt"]
        m_x, m_y = ecdf(m_tt / 60.0)
        t_x, t_y = ecdf(t_tt / 60.0)
        x_cap = max(np.percentile(m_x, 99.0), np.percentile(t_x, 99.0))
        ax.plot(t_x, t_y, color=th["torchdnl"], drawstyle="steps-post", zorder=3,
                linewidth=HOUSE_STROKE["torchdnl_gpu"]["lw"],
                linestyle=HOUSE_STROKE["torchdnl_gpu"]["ls"])
        ax.plot(m_x, m_y, color=th["matsim"], drawstyle="steps-post", zorder=4,
                linewidth=HOUSE_STROKE["matsim"]["lw"],
                linestyle=HOUSE_STROKE["matsim"]["ls"])
        ax.set_xlim(0, x_cap)
        ax.set_ylim(0, 1.001)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        w = distributional.pooled_wasserstein(t_tt, m_tt)
        w_txt = f"{w:.2f}" if w < 10 else f"{w:.1f}"
        _stat_box(ax, [f"$W = {w_txt}$ s",
                       f"{len(t_tt) // sc['n_seeds']:,} trips/seed"], th, "lower right")
        ax.set_xlabel("Completed-trip travel time (min)", fontsize=7)
        if col == 0:
            ax.set_ylabel("Cum. fraction of trips", fontsize=7.5)

    handles = [
        mlines.Line2D([], [], color=th["torchdnl"],
                      linewidth=HOUSE_STROKE["torchdnl_gpu"]["lw"],
                      linestyle=HOUSE_STROKE["torchdnl_gpu"]["ls"],
                      label="TorchDNL"),
        mlines.Line2D([], [], color=th["matsim"],
                      linewidth=HOUSE_STROKE["matsim"]["lw"],
                      linestyle=HOUSE_STROKE["matsim"]["ls"], label="MATSim"),
        mlines.Line2D([], [], color=th["base"], linewidth=1.0, label="$y = x$"),
    ]
    house_legend(fig, handles, th, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                 ncol=3, handlelength=2.2, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.9, w_pad=1.4)

    if dark:
        fig.patch.set_alpha(0.0)
        fig.savefig(output_path, format="png", dpi=400, transparent=True,
                    bbox_inches="tight", pad_inches=0.04)
    else:
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 6: single-column fidelity strip for the MAIN paper
# --------------------------------------------------------------------------
def plot_fidelity_strip(scenarios: list[dict], output_path: str,
                        dark: bool = False) -> None:
    """The main-paper counterpart of `plot_combined_scenario_grid`: one row of
    en-route curves, one panel per metropolitan scenario, sized for a single
    `\\columnwidth` slot.

    The grid version needs three rows because the appendix argues the case
    from scratch. Here the argument is already made in prose, so the figure
    only has to show the one thing prose cannot: that the two engines trace
    the same day, bin for bin, over a full 36 h horizon. The per-link and
    per-trip agreement is carried as an annotation rather than as two more
    rows of panels — the numbers are the claim, the plots of them are not.

    Each element of `scenarios` is a dict with keys: `label`, `n_links`,
    `n_seeds`, `bins`, `matsim_en_route_med`, `torchdnl_en_route_med`,
    `en_route_geh_max`, `pearson_r`, `wasserstein_s`.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    th = _theme(dark)
    n_cols = len(scenarios)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.33, 1.62))
    axes = np.atleast_1d(axes)

    for col, sc in enumerate(scenarios):
        ax = axes[col]
        _style_panel(ax, th)
        m_med = np.asarray(sc["matsim_en_route_med"], dtype=float)
        t_med = np.asarray(sc["torchdnl_en_route_med"], dtype=float)
        x = np.asarray(sc["bins"], dtype=float)[:-1] / 3600.0

        # Same clipping rule as the appendix grid: the caches run to 36 h so
        # the demand can drain, and the empty tail is half the axis otherwise.
        loaded = np.flatnonzero((m_med > 0) | (t_med > 0))
        lo, hi = (x[loaded[0]], x[min(loaded[-1] + 2, len(x) - 1)]) if loaded.size else (x[0], x[-1])
        lo = max(0.0, lo - 0.5)

        ax.step(x, t_med, where="post", color=th["torchdnl"], zorder=3,
                linewidth=HOUSE_STROKE["torchdnl_gpu"]["lw"] * 0.75,
                linestyle=HOUSE_STROKE["torchdnl_gpu"]["ls"],
                solid_joinstyle="round")
        ax.step(x, m_med, where="post", color=th["matsim"], zorder=4,
                linewidth=HOUSE_STROKE["matsim"]["lw"],
                linestyle=HOUSE_STROKE["matsim"]["ls"])

        ax.set_xlim(lo, hi)
        peak = max(m_med.max(), t_med.max())
        # Headroom for the three-line annotation, which sits over the empty
        # pre-dawn corner rather than over the peak.
        ax.set_ylim(0, peak * 1.62)
        ax.xaxis.set_major_locator(plt.MultipleLocator(6 if hi - lo >= 18 else 4))
        yticks = plt.MaxNLocator(nbins=3).tick_values(0, peak)
        ax.set_yticks([t for t in yticks if 0 <= t <= peak * 1.02])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _si(v)))
        ax.tick_params(labelsize=6)

        _stat_box(ax, [rf"en route: max GEH {sc['en_route_geh_max']:.2f}",
                       rf"link volume: $r = {sc['pearson_r']:.5f}$",
                       rf"trip time: $W = {sc['wasserstein_s']:.1f}$ s"],
                  th, "upper left", fontsize=5.4)

        ax.set_title(sc["label"], fontsize=7.5, color=th["ink"], pad=9,
                     fontweight="bold")
        ax.annotate(f"{sc['n_links']:,} links · {sc['n_seeds']} seeds",
                    xy=(0.5, 1.015), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=5.2, color=th["sub"])
        ax.set_xlabel("Hour of day", fontsize=6.5, labelpad=1.5)
        if col == 0:
            ax.set_ylabel("Agents en route", fontsize=6.5, labelpad=1.5)

    handles = [
        mlines.Line2D([], [], color=th["torchdnl"],
                      linewidth=HOUSE_STROKE["torchdnl_gpu"]["lw"] * 0.75,
                      linestyle=HOUSE_STROKE["torchdnl_gpu"]["ls"],
                      label="TorchDNL"),
        mlines.Line2D([], [], color=th["matsim"],
                      linewidth=HOUSE_STROKE["matsim"]["lw"],
                      linestyle=HOUSE_STROKE["matsim"]["ls"], label="MATSim"),
    ]
    house_legend(fig, handles, th, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                 ncol=2, handlelength=2.0, columnspacing=1.6, fontsize=6.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)

    if dark:
        fig.patch.set_alpha(0.0)
        fig.savefig(output_path, format="png", dpi=400, transparent=True,
                    bbox_inches="tight", pad_inches=0.03)
    else:
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
