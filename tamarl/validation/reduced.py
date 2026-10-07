"""Load the compact per-seed form written by scripts/reduce_validation_events.py.

Both fidelity entry points -- scripts/validate_against_matsim.py for the numbers
and scripts/make_validation_figures.py for the appendix figure cache -- need the
same per-bin series and link volumes, and both originally obtained them by parsing
every event of every seed into a list. That is fine for Sioux Falls (131,516 trips
on 3,968 links) and impossible for Berlin, where one seed is 52.3 million events
and the loops hold all twenty seeds at once.

Keeping this in one place rather than copied into both scripts matters because the
binning contract -- bucket width and horizon -- has to agree with how the reduction
was produced, and two copies of that check would drift.
"""

from __future__ import annotations

import os

import numpy as np

REQUIRED = ("bins", "departures", "arrivals", "stuck", "en_route", "link_volumes")


def load_reduced(seed_dir: str, max_steps: int, bucket_sec: int) -> dict | None:
    """Return the reduced series for one seed, or None if it was never reduced.

    None lets the caller fall back to the full-event path, so the scenarios whose
    numbers are already published keep being computed exactly as they were.
    """
    path = os.path.join(seed_dir, "reduced.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    missing = [k for k in REQUIRED if k not in z]
    if missing:
        raise SystemExit(f"ABORT: {path} lacks {missing}; re-run the reduction")
    stored_bucket, stored_steps = int(z["bucket_sec"]), int(z["max_steps"])
    if (stored_bucket, stored_steps) != (bucket_sec, max_steps):
        raise SystemExit(
            f"ABORT: {path} was reduced at bucket_sec={stored_bucket}, "
            f"max_steps={stored_steps}, but this run asks for "
            f"{bucket_sec}/{max_steps}. Re-reduce with matching settings rather "
            f"than comparing series binned differently."
        )
    return {k: z[k] for k in REQUIRED}


def load_pair(matsim_seed_dir: str, torchdnl_seed_dir: str, seed: int,
              max_steps: int, bucket_sec: int) -> tuple[dict, dict] | None:
    """Both sides' reduced series, or None if neither side was reduced.

    Refuses a one-sided reduction: comparing a reduced series against a freshly
    parsed one is how a silent binning mismatch would get into a published number.
    """
    m = load_reduced(matsim_seed_dir, max_steps, bucket_sec)
    t = load_reduced(torchdnl_seed_dir, max_steps, bucket_sec)
    if m is None and t is None:
        return None
    if (m is None) != (t is None):
        raise SystemExit(
            f"ABORT: seed {seed} is reduced on one side only "
            f"({'MATSim' if t is None else 'TorchDNL'} is not). Reduce both sides "
            f"or neither."
        )
    return m, t


def load_completed_leg_times(seed_dir: str) -> np.ndarray | None:
    """Travel times of COMPLETED TorchDNL legs, if they were extracted.

    MATSim's output_trips.csv.gz lists only trips that finished, while
    leg_travel_times.pt lists every leg including those still in the network at the
    horizon, whose times finalize_stuck_agents() imputes. Comparing the two directly
    is comparing different populations: on Los Angeles that is 18,601 extra samples
    out of 287,774 -- the longest ones -- and it put the Wasserstein distance at
    2,665 s where the completed sets agree to 0.66% on the mean.

    Returns None when the extraction was not run, so scenarios whose numbers are
    already published keep using every positive leg time exactly as before. On
    Berlin after the one-link-leg fix the two are the same set anyway, since every
    leg arrives.
    """
    path = os.path.join(seed_dir, "completed_leg_times.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)
