"""Reduce one validation seed's event stream to the handful of statistics the
fidelity comparison actually consumes, without ever holding the events in memory.

Why this exists: the appendix pipeline was written for Sioux Falls, where a seed's
events fit comfortably in a list. On Berlin one seed is 52.3 million events
(544 MB gzipped), and scripts/validate_against_matsim.py keeps every seed's events
AND derived arrays alive at once -- twenty seeds would be over a billion tuples.

Yet the comparison only ever needs:

  * per-bin counts of departures, arrivals and stuckAndAbort, from which the
    en-route series is the running difference, and
  * full-day per-link volumes, counted from `left link` events.

Everything else -- entered link, actend, actstart, vehicle enters/leaves traffic --
is discarded, and on these runs that is the large majority of the stream. Per-trip
travel times are not touched here at all: they come from MATSim's
output_trips.csv.gz and TorchDNL's leg_travel_times.pt, which are already compact.

The MATSim side is scanned line-by-line with a regex rather than through an XML
parser. MATSim writes exactly one <event .../> element per line, and a parser
costs several times more for no benefit at this volume; the scan verifies its own
coverage by counting `<event ` occurrences and reporting any it failed to read, so
a format change shows up as a loud mismatch instead of silently lost events.

    python scripts/reduce_validation_events.py \\
        --network .../berlin-v6.0-network-with-pt.xml \\
        --seed-dir results/validation_berlin/matsim/seed_0 --engine matsim \\
        --max-steps 129600 --bucket-sec 900

Writes <seed-dir>/reduced.npz. Idempotent: an existing file is kept unless --force.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

import numpy as np

# The only event names that carry information for this comparison, in both
# spellings: MATSim writes "left link" and "stuckAndAbort", TorchDNL's CSV writer
# uses EVENT_TYPE_NAMES, which spells the same things "left_link" and
# "stuckAndAbort".
WANTED = {
    "departure": "dep",
    "arrival": "arr",
    "stuckAndAbort": "stuck",
    "left link": "left",
    "left_link": "left",
}

_RE_TYPE = re.compile(r'\btype="([^"]*)"')
_RE_MODE = re.compile(r'\blegMode="([^"]*)"')
_RE_TIME = re.compile(r'\btime="([^"]*)"')
_RE_LINK = re.compile(r'\blink="([^"]*)"')


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, errors="replace")


class Accumulator:
    """Running per-bin time histograms and per-link volumes."""

    def __init__(self, max_steps: int, bucket_sec: int, num_links: int):
        self.bins = np.arange(0, max_steps + bucket_sec, bucket_sec, dtype=np.float64)
        n = len(self.bins) - 1
        self.bucket_sec = bucket_sec
        self.n_bins = n
        self.dep = np.zeros(n, dtype=np.int64)
        self.arr = np.zeros(n, dtype=np.int64)
        self.stuck = np.zeros(n, dtype=np.int64)
        self.link_vol = np.zeros(num_links, dtype=np.int64)
        self.n_read = 0
        self.n_outside = 0
        self.n_other_mode = 0

    def add_time(self, kind: str, t: float) -> None:
        # Mirror np.histogram's half-open binning; times at or past the horizon are
        # counted separately rather than folded into the last bin, so a horizon
        # mismatch between the two engines cannot hide inside it.
        idx = int(t // self.bucket_sec)
        if idx < 0 or idx >= self.n_bins:
            self.n_outside += 1
            return
        getattr(self, kind)[idx] += 1

    def add_link(self, edge_idx: int) -> None:
        if 0 <= edge_idx < len(self.link_vol):
            self.link_vol[edge_idx] += 1

    def en_route(self) -> np.ndarray:
        return np.cumsum(self.dep) - np.cumsum(self.arr) - np.cumsum(self.stuck)


def reduce_matsim(path: Path, acc: Accumulator, link_id_to_idx: dict[str, int],
                  leg_mode: str = "car") -> int:
    n_elements = 0
    want_mode = None if leg_mode == "any" else leg_mode
    with _open_text(path) as fh:
        for line in fh:
            if "<event " not in line:
                continue
            n_elements += line.count("<event ")
            m = _RE_TYPE.search(line)
            if m is None:
                continue
            kind = WANTED.get(m.group(1))
            if kind is None:
                acc.n_read += 1
                continue
            acc.n_read += 1
            if kind == "left":
                ml = _RE_LINK.search(line)
                if ml is not None:
                    idx = link_id_to_idx.get(ml.group(1))
                    if idx is not None:
                        acc.add_link(idx)
            else:
                if want_mode is not None:
                    mm = _RE_MODE.search(line)
                    if mm is None or mm.group(1) != want_mode:
                        acc.n_other_mode += 1
                        continue
                mt = _RE_TIME.search(line)
                if mt is not None:
                    acc.add_time(kind, float(mt.group(1)))
    return n_elements


def reduce_torchdnl(path: Path, acc: Accumulator, link_id_to_idx: dict[str, int]) -> int:
    n_rows = 0
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return 0
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            n_rows += 1
            acc.n_read += 1
            kind = WANTED.get(row[col["type"]])
            if kind is None:
                continue
            if kind == "left":
                idx = link_id_to_idx.get(row[col["link"]])
                if idx is not None:
                    acc.add_link(idx)
            else:
                acc.add_time(kind, float(row[col["time"]]))
    return n_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True)
    p.add_argument("--seed-dir", required=True)
    p.add_argument("--engine", required=True, choices=("matsim", "torchdnl"))
    p.add_argument("--max-steps", type=int, default=129600)
    p.add_argument("--bucket-sec", type=int, default=900)
    p.add_argument("--leg-mode", default="car",
                   help="Count only MATSim departure/arrival/stuckAndAbort events "
                        "whose legMode is this (default car); \"any\" disables the "
                        "filter. On Berlin, 851,977 of 1,267,499 departures are "
                        "teleported WALK access legs to and from the car, plus pt, "
                        "bike and ride -- 67%% of the stream. TorchDNL replays car "
                        "legs only, so counting the rest would compare two different "
                        "populations and bias the en-route envelope, which is the "
                        "appendix's headline metric. Link events need no filter: "
                        "every other mode is teleported, and the left-link count "
                        "matches the routes' consecutive-pair count exactly.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seed_dir = Path(args.seed_dir)
    out = seed_dir / "reduced.npz"
    if out.is_file() and not args.force:
        print(f"[skip] {out} already present")
        return 0

    src = (seed_dir / "output_events.xml.gz") if args.engine == "matsim" \
        else (seed_dir / "events.csv")
    if not src.is_file():
        print(f"ABORT: missing {src}", file=sys.stderr)
        return 1

    # Capacity scaling does not affect link INDEXING, so the default scale is fine
    # here -- this only needs the id -> index map, identical for both engines.
    from tamarl.envs.scenario_loader import parse_network

    _, _, link_id_to_idx, _ = parse_network(args.network)
    acc = Accumulator(args.max_steps, args.bucket_sec, len(link_id_to_idx))

    if args.engine == "matsim":
        n_elements = reduce_matsim(src, acc, link_id_to_idx, args.leg_mode)
        if n_elements != acc.n_read:
            print(f"ABORT: saw {n_elements:,} '<event ' occurrences but read "
                  f"{acc.n_read:,}; the events file is not one element per line "
                  f"and this scan would be dropping data", file=sys.stderr)
            return 1
    else:
        reduce_torchdnl(src, acc, link_id_to_idx)

    np.savez_compressed(
        out,
        bins=acc.bins,
        departures=acc.dep,
        arrivals=acc.arr,
        stuck=acc.stuck,
        en_route=acc.en_route(),
        link_volumes=acc.link_vol,
        n_events=np.int64(acc.n_read),
        n_outside_horizon=np.int64(acc.n_outside),
        n_other_mode=np.int64(acc.n_other_mode),
        leg_mode=str(args.leg_mode),
        bucket_sec=np.int64(args.bucket_sec),
        max_steps=np.int64(args.max_steps),
    )
    print(f"wrote {out}")
    print(f"  events read      : {acc.n_read:,}")
    print(f"  outside horizon  : {acc.n_outside:,}")
    print(f"  dropped, wrong mode: {acc.n_other_mode:,} (keeping legMode={args.leg_mode})")
    print(f"  departures       : {acc.dep.sum():,}")
    print(f"  arrivals         : {acc.arr.sum():,}")
    print(f"  stuckAndAbort    : {acc.stuck.sum():,}")
    print(f"  link traversals  : {acc.link_vol.sum():,} over "
          f"{int((acc.link_vol > 0).sum()):,} links")
    print(f"  peak en route    : {acc.en_route().max():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
