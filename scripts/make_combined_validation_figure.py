"""Render the single cross-scenario DNL-fidelity figure (3 fidelity signals x
N scenarios) for the AAMAS supplementary material, from the per-scenario `.npz`
caches written by `scripts/make_validation_figures.py --cache ...`.

Three signals, one per row: network-wide en-route count over the day (both
engines), full-day per-link volume parity with Pearson r, and the pooled
completed-trip travel-time ECDFs with the Wasserstein distance. The MATSim
inter-seed envelope and its inclusion rate are deliberately absent — with fixed
routes MATSim is near-deterministic, the band is ~0.1-0.2% wide, and the
inclusion rate measures the band's narrowness rather than the engines'
agreement (RAPPORT_AAMAS section 3).

As everywhere in this repo, the figure is emitted twice:

  <output>.pdf   light theme, for the paper / supplementary PDF
  <output>.png   TRANSPARENT background, light ink, for dark-background slides

Usage:
    python scripts/make_combined_validation_figure.py \\
        --cache "Sioux Falls=/path/validation_cache_sioux_falls.npz" \\
        --cache "Berlin=/path/validation_cache_berlin.npz" \\
        --cache "Los Angeles=/path/validation_cache_la.npz" \\
        --output aamas_2027_template/Figures/dnl_validation_combined_grid

Scenarios are plotted in the order given on the command line — pass them
smallest-to-largest so the grid reads left-to-right as a scale progression.
"""

from __future__ import annotations

import argparse

import numpy as np

from tamarl.validation import distributional
from tamarl.validation import geh as geh_mod
from tamarl.visualisation import validation_figures as vf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", action="append", required=True,
        help="label=path.npz, repeatable, in left-to-right column order.",
    )
    parser.add_argument("--output", required=True, help="Path stem; .pdf and .png are appended")
    args = parser.parse_args()

    usetex_ok = vf.setup_style()
    print(f"matplotlib text.usetex: {'enabled' if usetex_ok else 'disabled (STIX fallback)'}")

    scenarios = []
    for spec in args.cache:
        label, path = spec.split("=", 1)
        c = np.load(path)
        n_seeds = len(c["seeds"].tolist())
        sc = {
            "label": label,
            "n_links": int(c["matsim_link_vol"].shape[1]),
            "n_seeds": n_seeds,
            "bins": c["bins"],
            "matsim_en_route": c["matsim_en_route"],
            "torchdnl_en_route": c["torchdnl_en_route"],
            "matsim_link_vol": c["matsim_link_vol"],
            "torchdnl_link_vol": c["torchdnl_link_vol"],
            "matsim_trip_tt": c["matsim_trip_tt"],
            "torchdnl_trip_tt": c["torchdnl_trip_tt"],
        }
        scenarios.append(sc)

        # Echo every number the figure annotates, so the prose can quote them
        # without re-deriving anything.
        mv = sc["matsim_link_vol"].mean(axis=0)
        tv = sc["torchdnl_link_vol"].mean(axis=0)
        g_en = geh_mod.geh_series_from_counts(sc["torchdnl_en_route"], sc["matsim_en_route"])
        g_lv = geh_mod.geh_series_from_counts(sc["torchdnl_link_vol"], sc["matsim_link_vol"])
        w = distributional.pooled_wasserstein(sc["torchdnl_trip_tt"], sc["matsim_trip_tt"])
        print(
            f"\n{label}  ({path})\n"
            f"  seeds={n_seeds}  links={sc['n_links']:,}  "
            f"loaded links={int(((mv > 0) | (tv > 0)).sum()):,}\n"
            f"  trips/seed: MATSim {len(sc['matsim_trip_tt']) // n_seeds:,}  "
            f"TorchDNL {len(sc['torchdnl_trip_tt']) // n_seeds:,}\n"
            f"  mean trip tt: MATSim {sc['matsim_trip_tt'].mean():.1f} s  "
            f"TorchDNL {sc['torchdnl_trip_tt'].mean():.1f} s  "
            f"(gap {abs(sc['torchdnl_trip_tt'].mean() / sc['matsim_trip_tt'].mean() - 1) * 100:.2f}%)\n"
            f"  peak en route: {int(max(np.median(sc['matsim_en_route'], 0).max(), np.median(sc['torchdnl_en_route'], 0).max())):,}\n"
            f"  en-route  GEH: mean {g_en.mean():.3f}  max {g_en.max():.3f}  "
            f"%<5 {geh_mod.acceptance_rate(g_en) * 100:.1f}\n"
            f"  link-vol  GEH: mean {g_lv.mean():.3f}  max {g_lv.max():.3f}  "
            f"%<5 {geh_mod.acceptance_rate(g_lv) * 100:.1f}  (n={g_lv.size:,})\n"
            f"  Pearson r (all links): {np.corrcoef(mv, tv)[0, 1]:.6f}\n"
            f"  Wasserstein: {w:.3f} s"
        )

    stem = args.output[:-4] if args.output.endswith((".pdf", ".png")) else args.output
    vf.plot_combined_scenario_grid(scenarios, stem + ".pdf", dark=False)
    print(f"\nWrote {stem}.pdf")
    vf.plot_combined_scenario_grid(scenarios, stem + ".png", dark=True)
    print(f"Wrote {stem}.png")


if __name__ == "__main__":
    main()
