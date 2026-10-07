"""One-off script: regenerate the 5 Leipzig-only ablation figures for the
rewritten "Additional Results and Ablations" appendix section.

Reads the merged top-level `*_final.csv` files from the (externally located)
ablation_bundle, NOT the raw per-seed sweep subdirectories. Ingolstadt is
intentionally excluded from every figure here.

No before/after MSA step-size comparison is produced (that investigation was
inconclusive and is not part of the paper).

Style: everything visual is inherited from the house style in
`tamarl.visualisation.ablation_figures`, which in turn takes its chrome from
`validation_figures.house_*` (verbatim from
`graphs/notebooks/performance_benchmark_graphs.ipynb` cell 0/2, the source of
the reference figure `combined_scalability.pdf`) and its per-algorithm colours
from `graphs/notebooks/learning_graphs.ipynb` cell 3 `algo_config` (the source
of the main paper's `learning_and_entropy_combined.pdf`). Nothing is re-picked
here.

Sizing: `supplementary_material.tex` places four of these at `\textwidth` =
6.3 in and `leipzig_msa_horizon` (two bars per panel) at `0.7\textwidth`; each
is DRAWN at its own placed width so LaTeX does not rescale it.

Colour: the two MSA-only figures (step-size schedules, training horizon) draw
every bar in MSA's own orange. Their categories are settings of ONE algorithm,
named on the x axis, so a per-bar hue would encode nothing and would clash with
the per-algorithm palette used by the other three figures.
Both themes are emitted for every figure (light `.pdf` for the paper,
transparent light-ink `.png` for slides).

Usage:
    TRAFFICGYM_ABLATION_BUNDLE=<bundle>/leipzig \
    TRAFFICGYM_FIGURE_OUT=<outdir> \
    python scripts/generate_leipzig_appendix_figures.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from tamarl.visualisation.ablation_figures import (
    plot_categorical_bars,
    plot_metric_vs_param,
    setup_style,
)

BUNDLE = Path(os.environ.get("TRAFFICGYM_ABLATION_BUNDLE", "ablation_bundle/leipzig"))
OUT = Path(os.environ.get("TRAFFICGYM_FIGURE_OUT", "figures/ablation_leipzig"))
OUT.mkdir(parents=True, exist_ok=True)

# The supplementary is a single-column document with a 6.3 in text width and
# places each of these at \textwidth. Draw at exactly that so nothing is
# resampled down on the page.
TEXT_WIDTH_IN = 6.3

# The horizon figure carries only two bars per panel, so it is placed at
# 0.7\textwidth instead and drawn to match.
HORIZON_WIDTH_IN = 0.7 * TEXT_WIDTH_IN


def load(name: str) -> list[dict]:
    with open(BUNDLE / name) as f:
        return list(csv.DictReader(f))


def to_float(rows: list[dict], cols: tuple[str, ...]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        for c in cols:
            if c in r and r[c] not in (None, ""):
                r[c] = float(r[c])
        out.append(r)
    return out


def main() -> None:
    usetex_ok = setup_style()
    print(f"text.usetex active: {usetex_ok}")

    metric_labels = {
        "mean_travel_time_final": "Mean Travel Time (s)",
        "mean_regret_final": "Mean Regret (s)",
    }
    written: list[str] = []

    # 1. Candidate-route count k sweep (ucb + msa), k in {3,5,9} -- k=15 missing.
    rows = to_float(load("A_k_sweep_final.csv"), ("top_k_paths", "mean_travel_time_final", "mean_regret_final"))
    written += plot_metric_vs_param(
        rows, param_col="top_k_paths",
        metrics=("mean_travel_time_final", "mean_regret_final"),
        output_path=str(OUT / "leipzig_k_sweep.pdf"),
        xlabel="Candidate route set size $k$",
        xticks=[3, 5, 9],
        figsize=(TEXT_WIDTH_IN, 2.25),
    )

    # 2. TD-oracle bin-width sensitivity (ucb + msa).
    rows = to_float(
        load("B_link_tt_interval_sweep_final.csv"),
        ("link_tt_interval", "mean_travel_time_final", "mean_regret_final"),
    )
    written += plot_metric_vs_param(
        rows, param_col="link_tt_interval",
        metrics=("mean_travel_time_final", "mean_regret_final"),
        output_path=str(OUT / "leipzig_oracle_bin_sweep.pdf"),
        xlabel="TD-oracle bin width (s)",
        xticks=[60, 150, 300, 600, 900],
        figsize=(TEXT_WIDTH_IN, 2.25),
    )

    # 3. MSA step-size schedule sensitivity -- flat categorical check, msa only.
    rows = to_float(load("B_msa_alpha_sweep_final.csv"), ("mean_travel_time_final", "mean_regret_final"))
    label_order = ["non_vanishing_control", "fast_power", "slow_power", "harmonic"]
    label_display = {
        "non_vanishing_control": "Non-vanishing (control)",
        "fast_power": "Fast power decay",
        "slow_power": "Slow power decay",
        "harmonic": "Harmonic ($1/n$)",
    }
    written += plot_categorical_bars(
        rows, label_col="label", label_order=label_order, label_display=label_display,
        metrics=("mean_travel_time_final", "mean_regret_final"),
        metric_labels=metric_labels,
        output_path=str(OUT / "leipzig_msa_alpha_sweep.pdf"),
        figsize=(TEXT_WIDTH_IN, 2.35), rotation=28.0, agent="msa",
    )

    # 4. MSA training-horizon sensitivity (100 vs 300 episodes). ep300 has only seeds 1,2.
    rows = to_float(load("B_msa_horizon_final.csv"), ("mean_travel_time_final", "mean_regret_final"))
    written += plot_categorical_bars(
        rows, label_col="label", label_order=["ep100", "ep300"],
        label_display={"ep100": "100 episodes", "ep300": "300 episodes"},
        metrics=("mean_travel_time_final", "mean_regret_final"),
        metric_labels=metric_labels,
        output_path=str(OUT / "leipzig_msa_horizon.pdf"),
        figsize=(HORIZON_WIDTH_IN, 1.95), rotation=0.0, bar_width=0.5,
        agent="msa",
    )

    # 5. UCB exploration constant c sweep.
    rows = to_float(
        load("C_ucb_c_sweep_final.csv"),
        ("ucb_c", "mean_travel_time_final", "mean_regret_final", "epsilon_compliance_rate_final"),
    )
    written += plot_metric_vs_param(
        rows, param_col="ucb_c",
        metrics=("mean_travel_time_final", "mean_regret_final", "epsilon_compliance_rate_final"),
        output_path=str(OUT / "leipzig_ucb_c_sweep.pdf"),
        xlabel="UCB exploration constant $c$",
        log_x=True,
        figsize=(TEXT_WIDTH_IN, 2.1),
    )

    print("Done. Wrote:")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
