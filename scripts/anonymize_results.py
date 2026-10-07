"""Copy raw cluster result CSVs into the repository's `results/` tree, stripped of
anything that identifies the machine or the account that produced them.

The timing harness records which compute node and which Slurm job produced each cell.
Those are useful while the campaign runs and useless afterwards, but the node names
identify the cluster, so they cannot ship in a double-blind artifact. This script is
the only step between the cluster and `results/`, and it is committed so a reviewer
can see exactly what was removed.

What it does, per CSV:

- `hostname` -> a stable pseudonym (`node-01`, `node-02`, ...). The mapping is
  consistent across every file in one run, so "these two cells ran on the same node"
  survives, which is the only methodologically relevant part of the field.
- `slurm_job` -> a stable pseudonym (`job-001`, ...), same reasoning.
- `log` -> dropped. It held an absolute path on a node-local scratch disk that no
  longer exists.

Every other column is copied byte-for-byte. Directories are renamed on the way in
(see `LAYOUT`) so the result tree reads by scenario rather than by the campaign's
internal shorthand.

It also copies the fidelity validation's `summary.md` files (see `FIDELITY`), turns
their GEH tables into one tidy `validation/fidelity_geh.csv`, and merges every timing
cell into one tidy `benchmarks/all_cells.csv` so a reviewer can re-derive the paper's
tables from a single file.

    python scripts/anonymize_results.py --raw results/_raw --out results
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil

# raw campaign directory -> path under the published results tree.
# Anything not listed here is skipped (smoke tests, aborted probes).
LAYOUT = {
    # Primary wall-clock ladder: car-only populations, post-fix engine.
    "bench_berlin_car": "benchmarks/berlin",
    "bench_la_car": "benchmarks/los_angeles",
    "bench_sf36": "benchmarks/sioux_falls",
    # Superseded ladders, kept because the paper's provenance note refers to them.
    "bench_berlin36": "benchmarks/superseded/berlin_raw_prefix",
    "bench_la": "benchmarks/superseded/los_angeles_raw_prefix",
    "bench_berlin": "benchmarks/superseded/berlin_24h_fixed_capacity",
    "bench": "benchmarks/superseded/sioux_falls_24h_fixed_capacity",
    # Memory ceilings.
    "saturation": "memory/saturation",
    "vram": "memory/vram",
    # Learning-curve probes used to size the training campaign.
    "calib": "learning/calibration",
    "probe": "learning/probe_scale",
    "probe_conv": "learning/probe_convergence",
    "probe_top": "learning/probe_top_k",
    # RNG-determinism ablation.
    "ab_rng": "ablations/rng_seed_1",
    "ab_rng2": "ablations/rng_seed_2",
}

# raw campaign directory -> path under the published results tree, for the
# DNL-fidelity validation. Only the reduced report is published: the per-seed event
# dumps behind it are 50-60 GB per scenario.
FIDELITY = {
    "validation_berlin_B1car_fixed": "validation/berlin_1pct_car",
    "validation_los_angeles_L0p01car": "validation/los_angeles_1pct_car",
    "validation_berlin_B1car": "validation/berlin_1pct_car_prefix_engine",
}

DROP_COLUMNS = ("log",)
PSEUDONYM_COLUMNS = {"hostname": "node", "slurm_job": "job"}

# Columns of the merged timing table, in order. A cell that lacks one gets "".
MERGED_COLUMNS = (
    "ladder", "tag", "engine", "device", "scenario", "loading", "seed",
    "capacity_factor", "population_file", "n_agents", "arrived",
    "t_engine_s", "t_total_s", "t_setup_s", "steps_run", "threads",
    "hostname", "slurm_job", "cpu",
)


class Pseudonymizer:
    """Stable value -> `<prefix>-NN` mapping, assigned in sorted order of first sight."""

    def __init__(self) -> None:
        self._seen: dict[str, set[str]] = {}
        self._map: dict[str, dict[str, str]] = {}

    def observe(self, column: str, value: str) -> None:
        if value:
            self._seen.setdefault(column, set()).add(value)

    def freeze(self) -> None:
        for column, values in self._seen.items():
            prefix = PSEUDONYM_COLUMNS[column]
            width = max(2, len(str(len(values))))
            self._map[column] = {
                v: f"{prefix}-{i:0{width}d}"
                for i, v in enumerate(sorted(values), start=1)
            }

    def apply(self, column: str, value: str) -> str:
        if not value:
            return value
        return self._map.get(column, {}).get(value, value)

    def report(self) -> str:
        return ", ".join(
            f"{len(m)} distinct {c}" for c, m in sorted(self._map.items())
        )


def csv_files(raw: str) -> list[tuple[str, str]]:
    """(absolute source, destination relative to --out) for every CSV we publish."""
    found = []
    for src_dir, dst_dir in sorted(LAYOUT.items()):
        root = os.path.join(raw, src_dir)
        if not os.path.isdir(root):
            print(f"  skip (absent): {src_dir}")
            continue
        for name in sorted(os.listdir(root)):
            if name.endswith(".csv"):
                found.append((os.path.join(root, name), os.path.join(dst_dir, name)))
    return found


def rewrite(src: str, dst: str, pseudo: Pseudonymizer) -> None:
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        shutil.copyfile(src, dst)
        return
    fields = [f for f in rows[0] if f not in DROP_COLUMNS]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            out = {}
            for f in fields:
                v = r.get(f, "")
                out[f] = pseudo.apply(f, v) if f in PSEUDONYM_COLUMNS else v
            w.writerow(out)


def merge_cells(pairs: list[tuple[str, str]], out_root: str, pseudo: Pseudonymizer) -> int:
    """One tidy row per timing cell, across every ladder, into benchmarks/all_cells.csv.

    Adds two columns the per-cell files do not carry: `ladder` (which directory the
    cell came from, i.e. which protocol it belongs to) and `device`, because the CSV
    records the engine as `torchdnl` whether it ran on GPU or CPU and only the tag
    says which.
    """
    rows = []
    for src, rel in pairs:
        if not rel.startswith("benchmarks/") or not src.endswith("_bench.csv"):
            continue
        ladder = os.path.dirname(rel)[len("benchmarks/"):]
        with open(src, newline="") as fh:
            for r in csv.DictReader(fh):
                tag = r.get("tag", "")
                if r.get("engine") == "torchdnl":
                    device = "cpu" if "_torchdnl_cpu_" in tag else "gpu"
                else:
                    device = "cpu"
                out = {"ladder": ladder, "device": device}
                for c in MERGED_COLUMNS:
                    if c in out:
                        continue
                    v = r.get(c, "")
                    out[c] = pseudo.apply(c, v) if c in PSEUDONYM_COLUMNS else v
                rows.append(out)

    rows.sort(key=lambda r: (r["ladder"], r["scenario"], float(r["loading"] or 0),
                             r["engine"], r["device"], r["seed"]))
    dst = os.path.join(out_root, "benchmarks", "all_cells.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MERGED_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def copy_fidelity(raw: str, out_root: str) -> tuple[int, int]:
    """Copy each validation `summary.md` and flatten its GEH tables into one CSV."""
    copied, geh_rows = 0, []
    for src_dir, dst_dir in sorted(FIDELITY.items()):
        src = os.path.join(raw, src_dir, "report", "summary.md")
        if not os.path.exists(src):
            print(f"  skip (absent): {src_dir}/report/summary.md")
            continue
        dst = os.path.join(out_root, dst_dir, "summary.md")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        copied += 1

        label = os.path.basename(dst_dir)
        in_geh = False
        for line in open(src):
            line = line.strip()
            if line.startswith("## "):
                in_geh = line.startswith("## GEH")
                continue
            if not in_geh or not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] in ("series", "") or set(cells[0]) <= {"-"}:
                continue
            geh_rows.append(dict(zip(
                ("scenario", "series", "n", "mean", "median", "max",
                 "pct_geh_lt_5", "pct_geh_lt_10"),
                [label] + cells)))

    if geh_rows:
        dst = os.path.join(out_root, "validation", "fidelity_geh.csv")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(geh_rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(geh_rows)
    return copied, len(geh_rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw", required=True, help="Directory holding the campaign's result dirs.")
    p.add_argument("--out", required=True, help="Destination results tree.")
    args = p.parse_args()

    pairs = csv_files(args.raw)
    if not pairs:
        print("nothing to do")
        return 1

    # Two passes: collect every value to pseudonymize before writing anything, so the
    # mapping does not depend on the order files happen to be visited.
    pseudo = Pseudonymizer()
    for src, _ in pairs:
        with open(src, newline="") as fh:
            for r in csv.DictReader(fh):
                for column in PSEUDONYM_COLUMNS:
                    if column in r:
                        pseudo.observe(column, r[column])
    pseudo.freeze()

    for src, rel in pairs:
        rewrite(src, os.path.join(args.out, rel), pseudo)

    n_merged = merge_cells(pairs, args.out, pseudo)
    n_fid, n_geh = copy_fidelity(args.raw, args.out)

    print(f"wrote {len(pairs)} CSVs under {args.out}")
    print(f"merged {n_merged} timing cells into benchmarks/all_cells.csv")
    print(f"copied {n_fid} fidelity summaries, {n_geh} GEH rows")
    print(f"pseudonymized: {pseudo.report() or 'nothing'}")
    print(f"dropped columns: {', '.join(DROP_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
