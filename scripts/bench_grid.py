"""Grid definition and SLURM array-index mapper for the TorchDNL-vs-MATSim
wall-clock scaling benchmark (see scripts/bench_scaling.py for the protocol).

One array task = one (scenario, loading, engine, seed) cell. Arrays are submitted
PER ENGINE, because the two have wildly different walltimes and an array shares a
single --time: TorchDNL is step-bound and nearly flat in loading (Sioux Falls:
253 s at load 0.25, 513 s at load 3), while MATSim is event-bound and grows with
demand, so sizing one array to the other's worst case would either kill MATSim
cells or park TorchDNL cells in a scarcer partition for nothing.

Scenario notes
--------------
sioux_falls
    The loading ladder built by scripts/scale_population.py from the routed base
    population (65,758 persons). Dense: ~865 agents per link at load 1.0.

berlin
    202,939 links against 516,260 persons in the shipped 10% sample, i.e. ~2.5
    agents per link -- roughly 350x less densely loaded than Sioux Falls at load
    1.0. Since TorchDNL pays an O(links) floor every one of its 86,400 timesteps
    while MATSim pays per traversal event, the shipped 10% sample is expected to
    be a WEAK case for the GPU rather than a strong one. The ladder therefore
    continues past 10% by cloning the 10% sample, up to full-population
    equivalent (~5.16M), which is the regime the scaling claim is about.
"""

from __future__ import annotations

import argparse
import shlex
import sys

ENGINES = ("torchdnl", "torchdnl_cpu", "matsim", "sumo")

# loading -> population-file token produced by scripts/scale_population.py
SIOUX_LOADINGS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0)

# Berlin loadings are multiples of the ROUTED, EXPLICIT-DEPARTURE base, which is
# the shipped 10% sample filtered to the 108,430 agents both engines interpret
# identically (see scripts/filter_population_explicit_departures.py). So 0.1x
# stands in for Berlin 1%, 0.5x for 5%, 1x for 10%, and the rest push past the
# shipped sample to 3.47M agents. Files are named B<loading>_population.xml.
# 10.0 is the REAL city: the shipped sample is 10%, so loading 10 restores Berlin's
# full travelling population (1,085,000 car travellers) on the network at its true
# 100% capacity. Appended, not sorted in, so existing array indices keep meaning.
BERLIN_LOADINGS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 10.0)

# Los Angeles ships ROUTED populations at five sample sizes, each with a matching
# MATSim config whose capacity factors already equal the sample fraction. Loading
# is therefore expressed directly as that fraction, and capacity follows it.
# 1,175,146 links -- 5.8x Berlin's network.
# 0.02 and 0.03 are APPENDED, not inserted in sorted order, on purpose: the array
# index is this tuple's position, so inserting them would renumber every existing
# cell and a resubmitted index would silently run a different experiment. They
# exist to tighten the MATSim crossover bracket, which the five shipped samples
# leave at [74,009 ; 388,334] -- a factor of 5.2, wide enough that the
# interpolated crossover cannot be claimed against Berlin's factor-2.0 bracket.
# Subsampled from the 5% routed population, so their routes are a subset of one
# already validated against the SUMO network.
# 1.0 is likewise the REAL city: LA ships a 25% sample at most, so full Los Angeles
# is that sample cloned four times -- ~7.9M car travellers on the 100% network.
LA_LOADINGS = (0.001, 0.01, 0.05, 0.1, 0.25, 0.02, 0.03, 1.0)

SEEDS = (0, 1, 2)

# Capacity scales WITH the population, as MATSim's own downsampling convention
# does (a p% sample runs with flow/storageCapacityFactor = p). Varying demand
# against a fixed 100% network would sweep from a nearly empty road system to a
# gridlocked one, so the curve would mostly measure how congested the scenario
# happens to be. Holding V/C fixed instead leaves the number of events to process
# as the only variable, which is what the scaling claim is about.
#   berlin       base is the shipped 10% sample -> 0.1 at loading 1
#   sioux_falls  base is the scenario's native full population -> 1.0 at loading 1
CAPACITY_AT_UNIT_LOAD = {"berlin": 0.1, "sioux_falls": 1.0, "los_angeles": 1.0}


def _tag_num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def cells_for(scenario: str, engine: str) -> list[dict]:
    loadings = {
        "sioux_falls": SIOUX_LOADINGS,
        "berlin": BERLIN_LOADINGS,
        "los_angeles": LA_LOADINGS,
    }[scenario]
    prefix = {"berlin": "B", "los_angeles": "L", "sioux_falls": "S"}[scenario]
    out = []
    for load in loadings:
        for seed in SEEDS:
            out.append(
                {
                    "scenario": scenario,
                    "engine": engine,
                    "loading": load,
                    "seed": seed,
                    "capacity_factor": round(CAPACITY_AT_UNIT_LOAD[scenario] * load, 6),
                    "pop_tag": f"{prefix}{_tag_num(load)}",
                    "tag": f"{scenario}_L{_tag_num(load)}_{engine}_s{seed}",
                }
            )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True,
                   choices=("sioux_falls", "berlin", "los_angeles"))
    p.add_argument("--engine", required=True, choices=ENGINES)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int)
    g.add_argument("--count", action="store_true")
    g.add_argument("--list", action="store_true")
    args = p.parse_args()

    cells = cells_for(args.scenario, args.engine)

    if args.count:
        print(len(cells))
        return 0
    if args.list:
        for i, c in enumerate(cells):
            print(f"{i}\t{c['tag']}\tpop={c['pop_tag']}")
        return 0

    # Out of range raises rather than wrapping: a modular index would silently
    # reuse cell 0's tag and clobber a finished measurement.
    if args.index < 0 or args.index >= len(cells):
        print(f"ABORT: index {args.index} out of range (valid 0..{len(cells) - 1})", file=sys.stderr)
        return 1

    c = cells[args.index]
    for k in ("scenario", "engine", "loading", "seed", "capacity_factor",
              "pop_tag", "tag"):
        print(f"CELL_{k.upper()}={shlex.quote(str(c[k]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
