"""Run ONE cell of the controlled population-scale grid and write its own CSVs.

One SLURM array task calls this once. Each cell owns its output paths (derived
from its tag), so 270 concurrent tasks never append to the same file — that
would be a lost-update race with a silent loser. Merge at analysis time; every
row carries its own identifying fields.

Idempotent: if the cell's DONE sentinel already exists, the run is skipped. That
makes resubmitting a walltime-killed array index free for the cells that
already finished.

Metric computation is deliberately delegated to `scripts/run_ablation_sweep.py`
(`run_one`, `compute_h_global`) so that this campaign's numbers are produced by
exactly the same code path as the paper's existing ablation tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", required=True)
    p.add_argument("--population", required=True, help="Population file tag, e.g. S16")
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--scale_factor", type=float, required=True)
    p.add_argument("--tag", required=True, help="Unique cell identifier; names the output files")
    p.add_argument("--arm", default="", help="Experimental arm label; names nothing, only tags rows")
    p.add_argument("--s_level", type=float, default=0.0)
    # One parameter block per trip leg ("agent") or per OD pair ("od_pair"). The
    # population methods (msa, rd) are only themselves at od_pair: with one
    # vehicle per block MSA's flow averaging has nothing to average and RD's
    # update is the identity map.
    p.add_argument("--formulation", default="agent", choices=["agent", "od_pair", "centralized"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=86400)
    p.add_argument("--top_k_paths", type=int, default=9)
    p.add_argument("--window", type=int, default=10)
    # The TD-oracle time-bin width. It is NOT cosmetic: MSA's update target and
    # the Wardrop-regret metric are both read off this binning, so a rerun meant
    # to be comparable with an existing campaign must use that campaign's value
    # (the paper's JSON configs say 300.0; train()'s own default is 60.0).
    p.add_argument("--link_tt_interval", type=float, default=60.0)
    p.add_argument("--epsilon_ratio", type=float, default=0.10)
    # Cache the parsed ScenarioData next to the scenario. Worth it as soon as
    # more than one cell runs on the same population: parsing Berlin's 100%
    # plans is tens of minutes that every later cell then skips.
    p.add_argument("--save_pickle", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    final_csv = outdir / f"{args.tag}_final.csv"
    curves_csv = outdir / f"{args.tag}_curves.csv"
    done_flag = outdir / f"{args.tag}.DONE"

    if done_flag.exists():
        print(f"[skip] {args.tag} already complete ({done_flag})")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)

    # --- Gates: resolve and check everything cheap BEFORE importing torch ------
    scen = Path(args.scenario)
    if not scen.is_dir():
        print(f"ABORT: scenario dir not found: {scen}", file=sys.stderr)
        return 1
    pop_matches = [
        f for f in os.listdir(scen)
        if ("population" in f.lower() or "plans" in f.lower())
        and args.population in f.replace("-", "_").replace(".", "_").split("_")
    ]
    if not pop_matches:
        print(
            f"ABORT: no population file in {scen} matches token '{args.population}'. "
            f"Run scripts/scale_population.py first.",
            file=sys.stderr,
        )
        return 1
    print(f"[gate] population file: {pop_matches[0]}")

    from scripts.run_ablation_sweep import run_one  # noqa: E402  (after the cheap gates)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("ABORT: --device cuda but no GPU is visible to this job", file=sys.stderr)
            return 1
        print(f"[gate] GPU: {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")

    base_kwargs = dict(
        scenario_path=str(scen),
        population_filter=args.population,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        top_k_paths=args.top_k_paths,
        formulation=args.formulation,
        scale_factor=args.scale_factor,
        link_tt_interval=args.link_tt_interval,
        epsilon_ratio=args.epsilon_ratio,
        save_pickle=args.save_pickle,
    )

    print(
        f"[run] tag={args.tag} S={args.s_level} arm={args.arm} method={args.method} "
        f"formulation={args.formulation} seed={args.seed} "
        f"scale_factor={args.scale_factor} episodes={args.episodes}"
    )
    t0 = time.time()
    row, curves = run_one(base_kwargs, {"agent_type": args.method}, args.seed, args.window)
    wall = time.time() - t0

    ident = {
        "tag": args.tag,
        "scenario": scen.name,
        "S": args.s_level,
        "arm": args.arm,
        "method": args.method,
        "formulation": args.formulation,
        "seed": args.seed,
        "scale_factor": args.scale_factor,
        "population_tag": args.population,
        "episodes": args.episodes,
        "top_k_paths": args.top_k_paths,
        "link_tt_interval": args.link_tt_interval,
    }
    row = {**ident, **row, "cell_wall_time_s": wall}

    with open(final_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    if curves:
        curve_fields = list(ident.keys()) + [k for k in curves[0].keys() if k not in ident]
        with open(curves_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=curve_fields)
            w.writeheader()
            for c in curves:
                w.writerow({**ident, **c})

    done_flag.write_text(json.dumps({"wall_time_s": wall, **ident}, indent=2))
    print(f"[done] {args.tag} in {wall / 60:.1f} min -> {final_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
