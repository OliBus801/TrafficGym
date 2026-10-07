"""Publication-quality figures for the "Additional Results and Ablations"
appendix section, populated from CSVs produced by
`scripts/run_ablation_sweep.py` and by the external Leipzig ablation bundle.

HOUSE STYLE
-----------
Every value here is inherited, not invented. The chrome (spines, dashed
0.5 pt major grid on both axes, no minor ticks, frameless legend parked above
the panels, serif type) comes from `tamarl.visualisation.validation_figures`'s
`house_axes` / `house_legend` / `house_palette`, which are in turn verbatim
from the reference figure's own source,
`graphs/notebooks/performance_benchmark_graphs.ipynb` cell 0 (light) and
cell 2 (dark).

The per-ALGORITHM colours and markers below come from the main paper's
learning figure, `graphs/notebooks/learning_graphs.ipynb` cell 3
(`algo_config`), so UCB is the same purple here as it is there, MSA the same
orange, and so on. Nothing else in the repository is allowed to re-pick those.

Font sizes are deliberately NOT the notebook's 9-10 pt: the notebook draws at
6.5 in and lets LaTeX shrink the result, so its labels land on the page near
4 pt. `setup_style()`'s 7-8 pt is kept and every figure is drawn at its final
printed size instead.
"""

from __future__ import annotations

import numpy as np

from tamarl.visualisation.validation_figures import (
    HOUSE_DARK,
    HOUSE_GRID,
    HOUSE_LIGHT,
    house_axes,
    house_legend,
    house_palette,
    setup_style,
)

__all__ = [
    "setup_style",
    "house_axes",
    "house_legend",
    "house_palette",
    "finish",
    "AGENT_COLORS",
    "AGENT_COLORS_DARK",
    "AGENT_MARKERS",
    "AGENT_LABELS",
    "QUALITATIVE_CYCLE",
    "QUALITATIVE_CYCLE_DARK",
    "agent_color",
    "qualitative_cycle",
    "plot_metric_vs_param",
    "plot_categorical_bars",
    "plot_method_comparison_bars",
    "plot_learning_curves",
    "plot_link_tt_interval_sensitivity",
    "plot_exp3_heatmap",
]

# --------------------------------------------------------------------------
# Per-algorithm identity. Copied from `graphs/notebooks/learning_graphs.ipynb`
# cell 3 `algo_config`, which is what produced the main paper's
# `learning_and_entropy_combined.pdf`. Colour AND marker both separate the
# series, so the figures survive a greyscale print.
# --------------------------------------------------------------------------
AGENT_COLORS = {
    "random": "#7f7f7f",
    "aon": "#d62728",
    "msa": "#ff7f0e",
    "rd": "#8c564b",
    "epsilon_greedy": "#aec7e8",
    "exp3": "#1f77b4",
    "ucb": "#9467bd",
    "ts": "#2ca02c",
}

# Dark-slide counterparts: same identity, hues lifted just enough to hold
# contrast on a dark ground, following the notebook's own dark-cell recipe
# (#1f77b4 -> #58a6ff, #ff7f0e -> #ff9f43, #d62728 -> #ff6b6b).
AGENT_COLORS_DARK = {
    "random": "#b0b0b0",
    "aon": "#ff6b6b",
    "msa": "#ff9f43",
    "rd": "#c69183",
    "epsilon_greedy": "#aec7e8",
    "exp3": "#58a6ff",
    "ucb": "#bb8ae6",
    "ts": "#5fd35f",
}

AGENT_MARKERS = {
    "random": "o",
    "aon": "v",
    "msa": "^",
    "rd": ">",
    "epsilon_greedy": "s",
    "exp3": "d",
    "ucb": "*",
    "ts": "P",
}

AGENT_LABELS = {
    "ucb": "UCB",
    "msa": "MSA",
    "ts": "Thompson Sampling",
    "epsilon_greedy": r"$\epsilon$-Greedy",
    "exp3": "Exp3",
    "rd": "Replicator Dynamics",
    "aon": "AON",
    "random": "Random",
}

METRIC_LABELS = {
    "mean_travel_time_final": "Mean Travel Time (s)",
    "mean_regret_final": "Mean Regret (s)",
    "epsilon_compliance_rate_final": r"$\epsilon$-Compliance",
    "h_global": r"$H_{global}$ (nats)",
}

# For axes with no natural "algorithm" grouping (MSA step-size schedules,
# training horizons): the reference figure's own four engine hues, in its own
# draw order, which are already luminance-separated for greyscale.
QUALITATIVE_CYCLE = [
    HOUSE_LIGHT["torchdnl_gpu"],
    HOUSE_LIGHT["matsim"],
    HOUSE_LIGHT["sumo"],
    HOUSE_LIGHT["torchdnl_cpu"],
]
QUALITATIVE_CYCLE_DARK = [
    HOUSE_DARK["torchdnl_gpu"],
    HOUSE_DARK["matsim"],
    HOUSE_DARK["sumo"],
    HOUSE_DARK["torchdnl_cpu"],
]


def agent_color(agent: str, dark: bool, fallback_index: int = 0) -> str:
    """The one colour this algorithm is allowed to have, light or dark."""
    table = AGENT_COLORS_DARK if dark else AGENT_COLORS
    cycle = qualitative_cycle(dark)
    return table.get(agent, cycle[fallback_index % len(cycle)])


def qualitative_cycle(dark: bool) -> list:
    return list(QUALITATIVE_CYCLE_DARK if dark else QUALITATIVE_CYCLE)


def finish(fig, out_stem: str, dark: bool) -> str:
    """Emit one theme of a figure.

    Hard project convention: every figure ships twice, a light PDF for the
    paper and a transparent light-ink PNG for dark slides. Same geometry, same
    numbers, different ink. Mirrors `scripts/make_scalability_figures.finish`.
    """
    import matplotlib.pyplot as plt

    path = f"{out_stem}.{'png' if dark else 'pdf'}"
    if dark:
        # Transparent so the deck's ground shows through; the axes patch has
        # to be cleared too or matplotlib paints white under the data.
        fig.patch.set_alpha(0.0)
        for ax in fig.axes:
            ax.patch.set_alpha(0.0)
        fig.savefig(path, format="png", dpi=400, transparent=True,
                    bbox_inches="tight", pad_inches=0.03)
    else:
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


def _agg(df, group_cols, value_col):
    """Mean +/- std of `value_col` grouped by `group_cols` (pandas-free)."""
    groups: dict[tuple, list[float]] = {}
    for row in df:
        key = tuple(row[c] for c in group_cols)
        if value_col in row and row[value_col] not in (None, ""):
            groups.setdefault(key, []).append(float(row[value_col]))
    out = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = (float(np.mean(arr)), float(np.std(arr)))
    return out


# --------------------------------------------------------------------------
# A2 / C: metric vs. a scalar hyperparameter, one panel per metric, one line
# per agent (or a single line if the axis has a single agent).
#
# Stroke geometry follows the reference figure: a solid 1.3 pt line carrying
# the algorithm's own colour and marker, errorbars in the same ink, legend as
# one frameless row above the panels.
# --------------------------------------------------------------------------
def plot_metric_vs_param(
    rows: list[dict],
    param_col: str,
    metrics: tuple[str, ...],
    output_path: str,
    xlabel: str,
    log_x: bool = False,
    xticks: list | None = None,
    figsize: tuple[float, float] | None = None,
    themes: tuple[bool, ...] = (False, True),
) -> list[str]:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    agents = sorted({r["agent_type"] for r in rows})
    n = len(metrics)
    stem = output_path[:-4] if output_path.endswith(".pdf") else output_path
    size = figsize or (2.6 * n, 2.3)
    written = []

    for dark in themes:
        pal = house_palette(dark)
        fig, axes = plt.subplots(1, n, figsize=size)
        if n == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            house_axes(ax, pal, x_grid=True, y_grid=True)
            for i, agent in enumerate(agents):
                sub = [r for r in rows if r["agent_type"] == agent]
                agg = _agg(sub, (param_col,), metric)
                if not agg:
                    continue
                xs = sorted(agg.keys(), key=lambda k: k[0])
                x_vals = [k[0] for k in xs]
                y_mean = [agg[k][0] for k in xs]
                y_std = [agg[k][1] for k in xs]
                color = agent_color(agent, dark, i)
                ax.errorbar(
                    x_vals, y_mean, yerr=y_std, color=color,
                    marker=AGENT_MARKERS.get(agent, "o"),
                    markersize=5.0 if AGENT_MARKERS.get(agent) in ("*", "P") else 4.0,
                    linewidth=1.4, capsize=2, capthick=0.8, elinewidth=0.8,
                    zorder=3 + i, label=AGENT_LABELS.get(agent, agent),
                )
            if log_x:
                ax.set_xscale("log")
                # Minor ticks are suppressed by house_axes; a log axis still
                # needs its decade ticks back.
                import matplotlib.ticker as mticker
                ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
            elif xticks is not None:
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(v) for v in xticks])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))

        fig.tight_layout()
        if len(agents) > 1:
            handles = [
                mlines.Line2D(
                    [], [], color=agent_color(a, dark, i),
                    marker=AGENT_MARKERS.get(a, "o"),
                    markersize=5.0 if AGENT_MARKERS.get(a) in ("*", "P") else 4.0,
                    linewidth=1.4, label=AGENT_LABELS.get(a, a),
                )
                for i, a in enumerate(agents)
            ]
            house_legend(fig, handles, pal, loc="lower center",
                         bbox_to_anchor=(0.5, 1.0), ncol=len(agents))
        fig.align_xlabels(axes)
        written.append(finish(fig, stem, dark))
    return written


# --------------------------------------------------------------------------
# Categorical sweep axis (MSA step-size schedules, training horizon): grouped
# bars, one panel per metric, single agent. Same chrome as the line panels so
# the two read as one family.
# --------------------------------------------------------------------------
def plot_categorical_bars(
    rows: list[dict],
    label_col: str,
    label_order: list[str],
    label_display: dict[str, str],
    metrics: tuple[str, ...],
    output_path: str,
    metric_labels: dict[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    bar_width: float = 0.62,
    rotation: float = 22.0,
    agent: str | None = None,
    themes: tuple[bool, ...] = (False, True),
) -> list[str]:
    """Grouped bars over a categorical sweep axis.

    `agent`: when every bar is the SAME algorithm at different settings (MSA
    step-size schedules, MSA training horizons), pass its key and all bars take
    that algorithm's one colour. Giving each bar a different hue there would
    encode nothing -- the x axis already names the settings -- and would
    contradict the per-algorithm palette, since only one of the cycle's hues is
    the algorithm's own. Leave it None only when the categories really are
    different things.
    """
    import matplotlib.pyplot as plt

    metric_labels = {**METRIC_LABELS, **(metric_labels or {})}
    n = len(metrics)
    stem = output_path[:-4] if str(output_path).endswith(".pdf") else str(output_path)
    size = figsize or (2.6 * n, 2.3)
    x = np.arange(len(label_order))
    written = []

    for dark in themes:
        pal = house_palette(dark)
        cycle = qualitative_cycle(dark)
        fig, axes = plt.subplots(1, n, figsize=size)
        if n == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            # Bars read against a horizontal rule only; a vertical grid behind
            # them is noise.
            house_axes(ax, pal, x_grid=False, y_grid=True)
            means, stds = [], []
            for lab in label_order:
                sub = [r for r in rows if r[label_col] == lab]
                agg = _agg(sub, (), metric)
                m, s = next(iter(agg.values())) if agg else (0.0, 0.0)
                means.append(m)
                stds.append(s)
            if agent is not None:
                colors = [agent_color(agent, dark)] * len(label_order)
            else:
                colors = [cycle[i % len(cycle)] for i in range(len(label_order))]
            ax.bar(
                x, means, bar_width, yerr=stds, color=colors, capsize=2.5,
                zorder=3, edgecolor=pal["box"], linewidth=0.5,
                error_kw={"ecolor": pal["sub"], "elinewidth": 0.8,
                          "capthick": 0.8, "zorder": 4},
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [label_display.get(lab, lab) for lab in label_order],
                rotation=rotation, ha="right" if rotation else "center",
                rotation_mode="anchor" if rotation else None,
            )
            ax.set_ylabel(metric_labels.get(metric, metric))
            lo = min(means[i] - stds[i] for i in range(len(means)))
            hi = max(means[i] + stds[i] for i in range(len(means)))
            pad = max((hi - lo) * 0.8, hi * 0.01)
            ax.set_ylim(max(0, lo - pad), hi + pad)
            ax.set_xlim(-0.6, len(label_order) - 0.4)

        fig.tight_layout()
        written.append(finish(fig, stem, dark))
    return written


# --------------------------------------------------------------------------
# A3: candidate-route method comparison (grouped bars).
# --------------------------------------------------------------------------
def plot_method_comparison_bars(
    rows: list[dict],
    metric: str,
    output_path: str,
    methods: tuple[str, ...] = ("penalty", "yen"),
    themes: tuple[bool, ...] = (False, True),
) -> list[str]:
    import matplotlib.pyplot as plt

    agents = sorted({r["agent_type"] for r in rows})
    x = np.arange(len(agents))
    width = 0.8 / len(methods)
    stem = output_path[:-4] if output_path.endswith(".pdf") else output_path
    written = []

    for dark in themes:
        pal = house_palette(dark)
        cycle = qualitative_cycle(dark)
        fig, ax = plt.subplots(figsize=(2.6 + 0.6 * len(agents), 2.4))
        house_axes(ax, pal, x_grid=False, y_grid=True)

        for i, method in enumerate(methods):
            means, stds = [], []
            for agent in agents:
                sub = [r for r in rows
                       if r["agent_type"] == agent and r.get("path_method") == method]
                agg = _agg(sub, (), metric)
                m, s = next(iter(agg.values())) if agg else (0.0, 0.0)
                means.append(m)
                stds.append(s)
            offset = (i - (len(methods) - 1) / 2) * width
            ax.bar(
                x + offset, means, width, yerr=stds, color=cycle[i % len(cycle)],
                capsize=2, label=method.capitalize(), zorder=3,
                edgecolor=pal["box"], linewidth=0.4,
                error_kw={"ecolor": pal["sub"], "elinewidth": 0.8,
                          "capthick": 0.8, "zorder": 4},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS.get(a, a) for a in agents])
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        handles, _ = ax.get_legend_handles_labels()
        house_legend(ax, handles, pal, loc="upper right", handlelength=1.4)

        fig.tight_layout()
        written.append(finish(fig, stem, dark))
    return written


# --------------------------------------------------------------------------
# B1 / B2: learning curves (MTT vs. episode), one line per group.
# --------------------------------------------------------------------------
def plot_learning_curves(
    curve_rows: list[dict],
    group_col: str,
    output_path: str,
    metric: str = "mean_travel_time",
    ylabel: str = "Mean Travel Time (s)",
    group_labels: dict | None = None,
    themes: tuple[bool, ...] = (False, True),
) -> list[str]:
    import matplotlib.pyplot as plt

    groups = sorted({r[group_col] for r in curve_rows})
    stem = output_path[:-4] if output_path.endswith(".pdf") else output_path
    written = []

    for dark in themes:
        pal = house_palette(dark)
        cycle = qualitative_cycle(dark)
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        house_axes(ax, pal, x_grid=True, y_grid=True)

        for i, group in enumerate(groups):
            sub = [r for r in curve_rows if r[group_col] == group]
            agg = _agg(sub, ("episode",), metric)
            if not agg:
                continue
            eps = sorted(agg.keys(), key=lambda k: k[0])
            label = (group_labels or {}).get(group, str(group))
            ax.plot([k[0] for k in eps], [agg[k][0] for k in eps],
                    color=cycle[i % len(cycle)], linewidth=1.3, zorder=3 + i,
                    label=label)

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        handles, _ = ax.get_legend_handles_labels()
        house_legend(ax, handles, pal, loc="upper right", handlelength=1.6,
                     fontsize=6.5, borderaxespad=0.3)

        fig.tight_layout()
        written.append(finish(fig, stem, dark))
    return written


# --------------------------------------------------------------------------
# B3: TD-oracle time-bin granularity sensitivity (MTT + Regret vs. bin width).
# --------------------------------------------------------------------------
def plot_link_tt_interval_sensitivity(rows: list[dict], output_path: str, **kw):
    return plot_metric_vs_param(
        rows,
        param_col="link_tt_interval",
        metrics=("mean_travel_time_final", "mean_regret_final"),
        output_path=output_path,
        xlabel="TD-oracle bin width (s)",
        **kw,
    )


# --------------------------------------------------------------------------
# C: Exp3 (eta, gamma) 2D grid heatmap.
# --------------------------------------------------------------------------
def plot_exp3_heatmap(rows: list[dict], metric: str, output_path: str,
                      themes: tuple[bool, ...] = (False, True)) -> list[str]:
    import matplotlib.pyplot as plt

    etas = sorted({r["exp3_eta"] for r in rows})
    gammas = sorted({r["exp3_gamma"] for r in rows})
    grid = np.full((len(gammas), len(etas)), np.nan)

    for r in rows:
        if metric not in r or r[metric] in (None, ""):
            continue
        grid[gammas.index(r["exp3_gamma"]), etas.index(r["exp3_eta"])] = float(r[metric])

    stem = output_path[:-4] if output_path.endswith(".pdf") else output_path
    written = []
    for dark in themes:
        pal = house_palette(dark)
        fig, ax = plt.subplots(figsize=(2.8, 2.4))
        im = ax.imshow(grid, cmap="magma_r", aspect="auto", origin="lower")
        ax.set_xticks(range(len(etas)))
        ax.set_xticklabels([str(e) for e in etas])
        ax.set_yticks(range(len(gammas)))
        ax.set_yticklabels([str(g) for g in gammas])
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\gamma$")
        ax.tick_params(colors=pal["faint"], labelcolor=pal["ink"])
        ax.xaxis.label.set_color(pal["ink"])
        ax.yaxis.label.set_color(pal["ink"])

        for i in range(len(gammas)):
            for j in range(len(etas)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if grid[i, j] > np.nanmean(grid) else "#0b0b0b")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(METRIC_LABELS.get(metric, metric), fontsize=7,
                       color=pal["ink"])
        cbar.ax.tick_params(labelsize=6, colors=pal["ink"])

        fig.tight_layout()
        written.append(finish(fig, stem, dark))
    return written
