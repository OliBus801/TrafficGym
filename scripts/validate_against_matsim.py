"""Orchestrate the TorchDNL-vs-MATSim DNL-fidelity validation for the paper
appendix: given N-seed reference runs from both simulators on the *same*
fixed Sioux Falls routes (see run_dnl_seed_sweep.py on the MATSim side and
run_torchdnl_fixed_routes.py on the TorchDNL side), compute:

  1. GEH statistic (replacing MSE/MAPE) on network-wide departures/arrivals/
     en-route-per-bin series, and on per-link full-day volumes.
  2. Whether TorchDNL falls inside MATSim's own inter-seed envelope for the
     en-route series, compared against MATSim's own leave-one-out
     self-consistency baseline (the headline "indistinguishable from one
     more MATSim seed" claim).
  3. Pooled travel-time distribution comparison (Wasserstein distance, KS
     test) between the two simulators.

Writes a summary markdown table and comparison figures to --output-dir.

Usage:
    python scripts/validate_against_matsim.py \\
        --network /path/to/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_network_PT.xml \\
        --matsim-root /path/to/matsim_project/output/siouxfalls-2014/dnl_validation \\
        --torchdnl-root tamarl/data/scenarios/sioux_falls/dnl_validation \\
        --output-dir tamarl/data/scenarios/sioux_falls/dnl_validation/report
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tamarl.envs.scenario_loader import parse_network
from tamarl.validation import distributional
from tamarl.validation import geh as geh_mod
from tamarl.validation.matsim_io import parse_matsim_events, parse_torchdnl_events_csv
from tamarl.validation.reduced import load_completed_leg_times, load_pair
from tamarl.visualisation.plot_histogram import _compute_leg_histogram_data

BUCKET_SEC_DEFAULT = 900  # 15 minutes, matching the granularity originally attempted


def _find_common_seeds(matsim_root: str, torchdnl_root: str) -> list[int]:
    matsim_seeds = {
        int(os.path.basename(p).removeprefix("seed_"))
        for p in glob.glob(os.path.join(matsim_root, "seed_*"))
        if os.path.exists(os.path.join(p, "output_events.xml.gz"))
        or os.path.exists(os.path.join(p, "reduced.npz"))
    }
    torchdnl_seeds = {
        int(os.path.basename(p).removeprefix("seed_"))
        for p in glob.glob(os.path.join(torchdnl_root, "seed_*"))
        if os.path.exists(os.path.join(p, "events.csv"))
        or os.path.exists(os.path.join(p, "reduced.npz"))
    }
    common = sorted(matsim_seeds & torchdnl_seeds)
    if not common:
        raise FileNotFoundError(
            f"No seeds with both a MATSim and a TorchDNL run found under "
            f"{matsim_root} / {torchdnl_root}."
        )
    return common


def _en_route_series(events: list[tuple], max_steps: int, bucket_sec: int) -> dict[str, np.ndarray]:
    bins, dep, arr, stuck, en_route = _compute_leg_histogram_data(
        events, max_steps, dt=1.0, bucket_size_sec=bucket_sec
    )
    return {"bins": bins, "departures": dep, "arrivals": arr, "stuck": stuck, "en_route": en_route}


def _link_volumes(events: list[tuple], num_links: int) -> np.ndarray:
    """Full-day per-link vehicle-entry counts, from left_link events (i.e. the
    link the vehicle is exiting — a standard link-volume definition)."""
    from tamarl.validation.matsim_io import EVT_LEFT_LINK

    counts = np.zeros(num_links, dtype=np.int64)
    for _, evt_type, _agent, edge_idx in events:
        if evt_type == EVT_LEFT_LINK and 0 <= edge_idx < num_links:
            counts[edge_idx] += 1
    return counts


def _matsim_trip_travel_times_sec(matsim_seed_dir: str) -> np.ndarray:
    trips_path = os.path.join(matsim_seed_dir, "output_trips.csv.gz")
    df = pd.read_csv(trips_path, sep=";")

    def hms_to_sec(s: str) -> float:
        h, m, sec = map(int, s.split(":"))
        return h * 3600 + m * 60 + sec

    return df["trav_time"].map(hms_to_sec).to_numpy(dtype=np.float64)


def _torchdnl_trip_travel_times_sec(torchdnl_seed_dir: str) -> np.ndarray:
    # Prefer the completed-leg extraction when it exists: MATSim's side lists only
    # finished trips, so including TorchDNL's imputed never-arrived legs would
    # compare two different populations. See load_completed_leg_times.
    completed = load_completed_leg_times(torchdnl_seed_dir)
    if completed is not None:
        return completed.astype(np.float64)
    leg_tt = torch.load(os.path.join(torchdnl_seed_dir, "leg_travel_times.pt"))
    leg_tt = leg_tt.numpy()
    valid = leg_tt > 0
    return leg_tt[valid]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--matsim-root", required=True, help="Directory containing seed_N/output_events.xml.gz etc.")
    parser.add_argument("--torchdnl-root", required=True, help="Directory containing seed_N/events.csv etc.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--bucket-sec", type=int, default=BUCKET_SEC_DEFAULT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _, _, link_id_to_idx, _ = parse_network(args.network)
    num_links = len(link_id_to_idx)

    seeds = _find_common_seeds(args.matsim_root, args.torchdnl_root)
    print(f"Comparing {len(seeds)} seeds: {seeds}")

    matsim_en_route, torchdnl_en_route = [], []
    matsim_dep, torchdnl_dep = [], []
    matsim_arr, torchdnl_arr = [], []
    matsim_link_vol, torchdnl_link_vol = [], []
    matsim_trip_tt, torchdnl_trip_tt = [], []
    bins = None

    for seed in seeds:
        matsim_seed_dir = os.path.join(args.matsim_root, f"seed_{seed}")
        torchdnl_seed_dir = os.path.join(args.torchdnl_root, f"seed_{seed}")

        pair = load_pair(matsim_seed_dir, torchdnl_seed_dir, seed,
                         args.max_steps, args.bucket_sec)
        if pair is not None:
            m_series, t_series = pair
            m_vol, t_vol = m_series["link_volumes"], t_series["link_volumes"]
            n_m = n_t = None
        else:
            matsim_events, _ = parse_matsim_events(
                os.path.join(matsim_seed_dir, "output_events.xml.gz"), link_id_to_idx
            )
            torchdnl_events = parse_torchdnl_events_csv(
                os.path.join(torchdnl_seed_dir, "events.csv"), link_id_to_idx
            )
            m_series = _en_route_series(matsim_events, args.max_steps, args.bucket_sec)
            t_series = _en_route_series(torchdnl_events, args.max_steps, args.bucket_sec)
            m_vol = _link_volumes(matsim_events, num_links)
            t_vol = _link_volumes(torchdnl_events, num_links)
            n_m, n_t = len(matsim_events), len(torchdnl_events)
        bins = m_series["bins"]

        matsim_en_route.append(m_series["en_route"])
        torchdnl_en_route.append(t_series["en_route"])
        matsim_dep.append(m_series["departures"])
        torchdnl_dep.append(t_series["departures"])
        matsim_arr.append(m_series["arrivals"])
        torchdnl_arr.append(t_series["arrivals"])

        matsim_link_vol.append(m_vol)
        torchdnl_link_vol.append(t_vol)

        matsim_trip_tt.append(_matsim_trip_travel_times_sec(matsim_seed_dir))
        torchdnl_trip_tt.append(_torchdnl_trip_travel_times_sec(torchdnl_seed_dir))

        if n_m is None:
            print(f"  seed {seed}: from reduced form "
                  f"(peak en route MATSim {m_series['en_route'].max():,}, "
                  f"TorchDNL {t_series['en_route'].max():,})")
        else:
            print(f"  seed {seed}: parsed ({n_m} MATSim events, {n_t} TorchDNL events)")

    matsim_en_route = np.stack(matsim_en_route)
    torchdnl_en_route = np.stack(torchdnl_en_route)
    matsim_dep = np.stack(matsim_dep)
    torchdnl_dep = np.stack(torchdnl_dep)
    matsim_arr = np.stack(matsim_arr)
    torchdnl_arr = np.stack(torchdnl_arr)
    matsim_link_vol = np.stack(matsim_link_vol)
    torchdnl_link_vol = np.stack(torchdnl_link_vol)
    matsim_trip_tt = np.concatenate(matsim_trip_tt)
    torchdnl_trip_tt = np.concatenate(torchdnl_trip_tt)

    # ---- 1. GEH on network-wide per-bin series ----
    geh_summaries = {}
    for name, model_seeds, ref_seeds in [
        ("en_route", torchdnl_en_route, matsim_en_route),
        ("departures", torchdnl_dep, matsim_dep),
        ("arrivals", torchdnl_arr, matsim_arr),
    ]:
        geh_vals = geh_mod.geh_series_from_counts(model_seeds, ref_seeds)
        geh_summaries[name] = geh_mod.summarize(geh_vals)

    # ---- GEH on per-link full-day volumes ----
    geh_summaries["link_volume"] = geh_mod.summarize(
        geh_mod.geh_series_from_counts(torchdnl_link_vol, matsim_link_vol)
    )

    # ---- 2. Envelope / self-consistency ----
    median, lower, upper = distributional.build_envelope(matsim_en_route)
    matsim_self_consistency = distributional.reference_self_consistency(matsim_en_route)
    torchdnl_inside_fraction = distributional.fraction_inside_envelope(torchdnl_en_route, lower, upper)

    # How WIDE the envelope is, not just how often a series falls inside it. An
    # inclusion rate is only interpretable next to this: on Berlin with fixed
    # routes MATSim is near-deterministic -- all 20 seeds produce identical
    # departure, arrival and link-traversal totals -- so its envelope is 0.2% wide
    # and a held-out seed still lands outside it 25% of the time. Reported alone,
    # that 75% reads as worse agreement than Sioux Falls' 86%, when in fact the
    # envelope there was far wider because congestion actually exercised the queue
    # model's stochasticity. Measured only on loaded bins: in a near-empty
    # overnight bin the relative width is noise divided by almost nothing.
    _busy = median > median.max() * 0.05
    _rel = np.divide(upper - lower, np.maximum(median, 1),
                     out=np.zeros(median.shape, dtype=float), where=median > 0)
    envelope_width = float(np.median(_rel[_busy])) if _busy.any() else float("nan")

    # ---- 3. Pooled travel-time distribution ----
    tt_wasserstein = distributional.pooled_wasserstein(torchdnl_trip_tt, matsim_trip_tt)
    tt_ks_stat, tt_ks_pvalue = distributional.ks_test(torchdnl_trip_tt, matsim_trip_tt)

    # ---- Report ----
    report_path = os.path.join(args.output_dir, "summary.md")
    with open(report_path, "w") as f:
        f.write("# TorchDNL vs MATSim — DNL-fidelity validation summary\n\n")
        f.write(f"Seeds compared: {seeds}\n\n")
        f.write("## GEH statistic (replaces MSE/MAPE)\n\n")
        f.write("| series | n | mean | median | max | % GEH<5 | % GEH<10 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, s in geh_summaries.items():
            f.write(
                f"| {name} | {s['n']} | {s['mean']:.2f} | {s['median']:.2f} | "
                f"{s['max']:.2f} | {s['pct_geh_lt_5']:.1%} | {s['pct_geh_lt_10']:.1%} |\n"
            )
        f.write("\n## Envelope / self-consistency (en-route series)\n\n")
        f.write(
            f"- MATSim leave-one-out self-consistency (fraction of a held-out MATSim "
            f"seed inside the envelope of the other MATSim seeds): "
            f"**{matsim_self_consistency:.1%}**\n"
        )
        f.write(
            f"- TorchDNL fraction inside MATSim's inter-seed envelope: "
            f"**{torchdnl_inside_fraction:.1%}**\n"
        )
        f.write(
            f"- MATSim envelope width on loaded bins (median, relative to the "
            f"median series): **{envelope_width:.1%}**. An inclusion rate is only "
            f"meaningful against this: the narrower the envelope, the harsher both "
            f"numbers above, so they must be compared across scenarios only "
            f"together.\n"
        )
        f.write("\n## Pooled trip travel-time distribution\n\n")
        f.write(f"- Wasserstein distance: **{tt_wasserstein:.2f}s**\n")
        f.write(f"- KS statistic: {tt_ks_stat:.4f} (p={tt_ks_pvalue:.3g})\n")
    print(f"Wrote {report_path}")

    # ---- Figure: en-route envelope vs TorchDNL ----
    x = bins[:-1] / 3600.0
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x, lower, upper, color="tab:blue", alpha=0.25, step="post", label="MATSim seed IQR envelope")
    ax.step(x, median, where="post", color="tab:blue", linewidth=2, label="MATSim median")
    for i, seed in enumerate(seeds):
        ax.step(
            x, torchdnl_en_route[i], where="post", color="tab:orange", alpha=0.5,
            linewidth=1, label="TorchDNL (per seed)" if i == 0 else None,
        )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Agents en route")
    ax.set_title("En-route agents per bin: TorchDNL vs MATSim seed envelope")
    ax.legend()
    fig_path = os.path.join(args.output_dir, "en_route_envelope.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {fig_path}")

    # ---- Figure: pooled travel-time distributions ----
    fig, ax = plt.subplots(figsize=(10, 6))
    bins_tt = np.linspace(0, max(matsim_trip_tt.max(), torchdnl_trip_tt.max()), 100)
    ax.hist(matsim_trip_tt, bins=bins_tt, density=True, alpha=0.5, label="MATSim")
    ax.hist(torchdnl_trip_tt, bins=bins_tt, density=True, alpha=0.5, label="TorchDNL")
    ax.set_xlabel("Trip travel time (s)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Pooled trip travel-time distribution "
        f"(Wasserstein={tt_wasserstein:.1f}s, KS p={tt_ks_pvalue:.3g})"
    )
    ax.legend()
    fig_path = os.path.join(args.output_dir, "travel_time_distribution.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
