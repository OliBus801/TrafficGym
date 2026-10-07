"""Render the single-column DNL-fidelity strip used in the MAIN paper
(Section 'Fidelity Against MATSim'), from the reduced `.npz` caches written by
`scripts/reduce_validation_cache.py`.

One row, one panel per metropolitan scenario: network-wide en-route count over
the simulated day, both engines. The per-link parity and per-trip travel-time
agreement are annotated as numbers rather than plotted — the full three-signal
grid stays in the supplementary material
(`scripts/make_combined_validation_figure.py`).

As everywhere in this repo, the figure is emitted twice:

  <output>.pdf   light theme, for the paper
  <output>.png   TRANSPARENT background, light ink, for dark-background slides

Usage:
    python scripts/make_fidelity_strip_figure.py \\
        --cache "Berlin=scratch/validation_strip/strip_berlin.npz" \\
        --cache "Los Angeles=scratch/validation_strip/strip_la.npz" \\
        --output aamas_2027_template/Figures/dnl_fidelity_strip

Scenarios are plotted in the order given on the command line.
"""

from __future__ import annotations

import argparse

import numpy as np

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
        sc = {
            "label": label,
            "n_links": int(c["n_links"]),
            "n_seeds": int(c["n_seeds"]),
            "bins": c["bins"],
            "matsim_en_route_med": c["matsim_en_route_med"],
            "torchdnl_en_route_med": c["torchdnl_en_route_med"],
            "en_route_geh_max": float(c["en_route_geh_max"]),
            "pearson_r": float(c["pearson_r"]),
            "wasserstein_s": float(c["wasserstein_s"]),
        }
        scenarios.append(sc)

        # Echo every number the figure annotates, so the prose can quote them
        # without re-deriving anything.
        print(
            f"\n{label}  ({path})\n"
            f"  seeds={sc['n_seeds']}  links={sc['n_links']:,}  "
            f"loaded links={int(c['n_loaded_links']):,}\n"
            f"  trips/seed: {int(c['trips_per_seed']):,}\n"
            f"  mean trip tt: MATSim {float(c['matsim_mean_tt']):.1f} s  "
            f"TorchDNL {float(c['torchdnl_mean_tt']):.1f} s  "
            f"(gap {abs(float(c['torchdnl_mean_tt']) / float(c['matsim_mean_tt']) - 1) * 100:.2f}%)\n"
            f"  en-route  GEH: mean {float(c['en_route_geh_mean']):.3f}  "
            f"max {sc['en_route_geh_max']:.3f}  %<5 {float(c['en_route_geh_pct_lt5']):.1f}\n"
            f"  link-vol  GEH: max {float(c['link_geh_max']):.3f}  "
            f"%<5 {float(c['link_geh_pct_lt5']):.1f}  (n={int(c['link_obs']):,})\n"
            f"  Pearson r: {sc['pearson_r']:.6f}\n"
            f"  Wasserstein: {sc['wasserstein_s']:.2f} s"
        )

    stem = args.output[:-4] if args.output.endswith((".pdf", ".png")) else args.output
    vf.plot_fidelity_strip(scenarios, stem + ".pdf", dark=False)
    print(f"\nWrote {stem}.pdf")
    vf.plot_fidelity_strip(scenarios, stem + ".png", dark=True)
    print(f"Wrote {stem}.png")


if __name__ == "__main__":
    main()
