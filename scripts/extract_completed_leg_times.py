"""Write out the travel times of COMPLETED TorchDNL legs only, for a fair
comparison against MATSim's trips file.

The fidelity report's Wasserstein distance compares two travel-time distributions,
and on Los Angeles they were not the same population. MATSim's
output_trips.csv.gz holds only trips that finished; TorchDNL's leg_travel_times.pt
holds every leg, including the ones still in the network when the 36 h horizon
expired, whose times finalize_stuck_agents() IMPUTES. On LA that is 18,601 extra
samples out of 287,774 -- the longest ones -- and it inflated the distance to
2,665 s. Truncating TorchDNL to MATSim's sample count drops its mean from 11,836 s
to 7,479 s, which is the size of the artefact.

Berlin never showed it because after the one-link-leg fix every Berlin leg
arrived, so there was nothing to impute and the two sets matched by construction.

Which legs completed is recoverable without re-simulating: TorchDNL emits one
`arrival` event per completed leg, in leg order, so an agent with k arrivals
completed its first k legs. One streaming pass over events.csv is enough.

    python scripts/extract_completed_leg_times.py \\
        --seed-dir results/validation_los_angeles_L0p01car/torchdnl/seed_0
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-dir", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    d = Path(args.seed_dir)
    out = d / "completed_leg_times.npy"
    if out.is_file() and not args.force:
        print(f"[skip] {out} already present")
        return 0
    events, legs_pt = d / "events.csv", d / "leg_travel_times.pt"
    for f in (events, legs_pt):
        if not f.is_file():
            print(f"ABORT: missing {f}", file=sys.stderr)
            return 1

    arrivals: Counter[int] = Counter()
    with open(events, newline="") as fh:
        r = csv.reader(fh)
        col = {k: i for i, k in enumerate(next(r))}
        ti, pi = col["type"], col["person"]
        for row in r:
            if row[ti] == "arrival":
                # Agent ids are written as "agent_<index>" by the runner.
                arrivals[int(row[pi].rsplit("_", 1)[1])] += 1

    leg_tt = torch.load(legs_pt).numpy()
    n_agents, max_legs = leg_tt.shape
    keep = np.zeros_like(leg_tt, dtype=bool)
    for agent, k in arrivals.items():
        if 0 <= agent < n_agents:
            keep[agent, : min(k, max_legs)] = True

    vals = leg_tt[keep & (leg_tt > 0)]
    np.save(out, vals)
    total = int((leg_tt > 0).sum())
    print(f"wrote {out}")
    print(f"  legs with a positive time : {total:,}")
    print(f"  of which COMPLETED        : {vals.size:,} "
          f"({100 * vals.size / max(total, 1):.1f}%)")
    print(f"  dropped (imputed, never arrived) : {total - vals.size:,}")
    print(f"  mean completed            : {vals.mean():,.1f}s "
          f"(all legs: {leg_tt[leg_tt > 0].mean():,.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
