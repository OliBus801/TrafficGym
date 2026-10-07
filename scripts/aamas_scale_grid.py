"""Grid definition and SLURM array-index mapper for the controlled
population-scale campaign (AAMAS 2027).

The scientific question: an earlier round of reviews objected that the UCB/TS advantage
was observed across networks that differ in topology, demand, capacity AND
population at once, so population scale was never isolated. This grid isolates
it on a SINGLE fixed network, and additionally separates "more agents" from
"more congestion" with two arms:

    scaleup   population xS, capacity x1  -> agent count up, V/C up
    isocong   population xS, capacity xS  -> agent count up, V/C constant

At S=1 the two arms are identical by construction, so only `scaleup` is
enumerated there.

Cells at one S level:  arms x methods x seeds  =  (1 or 2) x 6 x 5
Total across S in {1,2,4,8,16}:  30 + 4*60 = 270 cells per network.

One array task runs one cell. Walltime is set PER S LEVEL (one array
submission per S), because S is the cost axis of this very experiment: an array
shares a single --time across its tasks, so sizing every S to S=16's cost would
make the cheap levels queue for no reason.

Usage (from a batch script):
    ASSIGN="$(python3 scripts/aamas_scale_grid.py --s 16 --index $SLURM_ARRAY_TASK_ID)"
    eval "$ASSIGN"

Usage (to size an array):
    python3 scripts/aamas_scale_grid.py --s 16 --count     # -> 60
    python3 scripts/aamas_scale_grid.py --s 16 --list      # human-readable
"""

from __future__ import annotations

import argparse
import shlex
import sys

S_LEVELS = (1, 2, 4, 8, 16)
METHODS = ("ucb", "ts", "msa", "rd", "aon", "random")
SEEDS = (0, 1, 2, 3, 4)

# --- Loading probe -----------------------------------------------------------
# Calibration showed the doubling ladder above overshoots badly on Sioux Falls:
# arrival rate is 100% at S=1, 20.7% at S=4 and 1.7% at S=16, i.e. the network
# is in total gridlock well before the top of the ladder. At a 1.7% arrival rate
# almost every reward is imputed rather than experienced, so those cells would
# measure the imputation rule, not routing.
#
# The probe maps the arrival-rate curve between free flow and the onset of
# collapse so the real ladder can be placed where loading actually varies.
# `aon` brackets the worst case (all-or-nothing concentrates the whole demand on
# the shortest path) and `random` the best (it spreads demand), so together they
# give the envelope at each loading.
PROBE_LEVELS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
PROBE_METHODS = ("aon", "random")

# --- Convergence probe -------------------------------------------------------
# The loading probe runs ONE episode, so it cannot say where a *learned* policy
# ends up: at episode 1 UCB has no information and behaves like `random`, which
# collapses to a 19.3% arrival rate at load 3.0 while `aon` still delivers 84.8%.
# The usable top of the ladder therefore depends on whether learning recovers
# network throughput over an episode budget, which only a multi-episode run can
# answer. 20 episodes is enough to see the trend without paying for 100.
PROBE_CONV_LEVELS = (1.5, 2.0, 3.0)
PROBE_CONV_METHODS = ("ucb", "msa")

# --- Ladder-top probe --------------------------------------------------------
# The converge probe showed UCB's recovery happens between episodes ~8 and ~12:
# at load 3.0 it goes 26.5k -> 32.6k -> 9.9k seconds. The earlier 10-episode
# calibration that reported a 20.7% arrival rate at load 4.0 therefore sampled
# the exploration trough, not the converged policy, and cannot be used to rule
# that loading out. This probe re-tests the top of the ladder past the recovery.
PROBE_TOP_LEVELS = (4.0, 6.0)
PROBE_TOP_METHODS = ("ucb",)

PROBE_SETS = {
    "load": (PROBE_LEVELS, PROBE_METHODS),
    "converge": (PROBE_CONV_LEVELS, PROBE_CONV_METHODS),
    "top": (PROBE_TOP_LEVELS, PROBE_TOP_METHODS),
}


def probe_cells(which: str = "load") -> list[dict]:
    levels, methods = PROBE_SETS[which]
    out = []
    for level in levels:
        tag_level = f"{level:g}".replace(".", "p")
        for method in methods:
            out.append(
                {
                    "S": level,
                    "arm": f"probe-{which}",
                    "method": method,
                    "seed": 0,
                    "scale_factor": 1.0,
                    "pop_tag": f"S{tag_level}",
                    "tag": f"probe{'' if which == 'load' else which}_S{tag_level}_{method}",
                }
            )
    return out


def arms_for(s: int) -> tuple[str, ...]:
    """At S=1 the two arms coincide (capacity x1 == capacity xS), so don't run both."""
    return ("scaleup",) if s == 1 else ("scaleup", "isocong")


def cells_for(s: int) -> list[dict]:
    """Enumerate the cells at one S level, in a stable order.

    The order is part of the contract: an array index means a specific cell, so
    changing this ordering invalidates every already-written output path.
    """
    out = []
    for arm in arms_for(s):
        for method in METHODS:
            for seed in SEEDS:
                out.append(
                    {
                        "S": s,
                        "arm": arm,
                        "method": method,
                        "seed": seed,
                        # capacity scaling is what separates the two arms
                        "scale_factor": 1.0 if arm == "scaleup" else float(s),
                        # population file tag produced by scripts/scale_population.py
                        "pop_tag": f"S{s}",
                        "tag": f"S{s}_{arm}_{method}_seed{seed}",
                    }
                )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s", type=int, choices=S_LEVELS, help="Population scale level")
    p.add_argument(
        "--probe",
        choices=sorted(PROBE_SETS),
        help="Use a probe grid instead of the production grid",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="SLURM_ARRAY_TASK_ID -> emit shell assignments")
    g.add_argument("--count", action="store_true", help="Print the number of cells")
    g.add_argument("--list", action="store_true", help="Print every cell")
    args = p.parse_args()

    if args.probe:
        cells = probe_cells(args.probe)
    elif args.s is not None:
        cells = cells_for(args.s)
    else:
        print("ABORT: pass --s <level> or --probe", file=sys.stderr)
        return 1

    if args.count:
        print(len(cells))
        return 0

    if args.list:
        for i, c in enumerate(cells):
            print(f"{i}\t{c['tag']}\tscale_factor={c['scale_factor']}\tpop={c['pop_tag']}")
        return 0

    # Out of range must RAISE, never wrap: modular arithmetic would silently map
    # an oversized --array back onto cell 0's tag and clobber its output.
    if args.index < 0 or args.index >= len(cells):
        print(
            f"ABORT: index {args.index} out of range "
            f"(valid 0..{len(cells) - 1})",
            file=sys.stderr,
        )
        return 1

    c = cells[args.index]
    for key in ("S", "arm", "method", "seed", "scale_factor", "pop_tag", "tag"):
        print(f"CELL_{key.upper()}={shlex.quote(str(c[key]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
