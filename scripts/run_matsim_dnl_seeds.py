"""Produce the MATSim side of the TorchDNL-vs-MATSim DNL-fidelity validation:
one events file per seed, on fixed routes, in the layout
scripts/validate_against_matsim.py expects.

    <output-dir>/seed_<N>/output_events.xml.gz

This is the counterpart of scripts/run_torchdnl_fixed_routes.py. The two must be
given the SAME network, the SAME population and the SAME --capacity-factor, or
the comparison measures the configuration difference rather than the engines.

Why it reuses bench_scaling.build_matsim_config instead of writing its own config:
that function carries every correction this scenario needed -- the DOCTYPE MATSim
refuses to parse without, rebasing the template's relative input paths, reading the
CRS out of the network header, and REMOVING the transit and counts modules rather
than blanking their paths (a blank inputCountsFile resolves back onto the config
itself, and MATSim then reports a bogus 'Doctype "config_v2.dtd" not known').
Re-deriving that here would mean re-earning all of it.

The one deliberate difference from the timing harness: events are written. The
benchmark disables all output so neither engine is charged for I/O; validation
needs exactly that output, so these runs are NOT comparable as timings and their
wall-clock is not recorded.

Usage:
    python scripts/run_matsim_dnl_seeds.py \\
        --matsim-jar  $MATSIM_ROOT/matsim-example-project-...jar \\
        --matsim-config $MATSIM_ROOT/scenarios/berlin/berlin-v6.0.config.xml \\
        --matsim-runner org.matsim.project.OpenBerlinScenario \\
        --network   .../berlin-v6.0-network-with-pt.xml \\
        --population .../B1_population.xml \\
        --capacity-factor 0.1 \\
        --seeds 0 1 2 --output-dir .../dnl_validation_matsim
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_bench_scaling():
    """Import the sibling bench_scaling.py by path.

    scripts/ is not a package, so a plain `import bench_scaling` depends on the
    caller's cwd and would break under srun. Loading by explicit path does not.
    """
    spec = importlib.util.spec_from_file_location(
        "bench_scaling", _HERE / "bench_scaling.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_seed(args, bs, seed: int) -> Path:
    seed_dir = Path(args.output_dir).resolve() / f"seed_{seed}"
    events = seed_dir / "output_events.xml.gz"
    if events.is_file() and not args.force:
        print(f"[skip] seed {seed}: {events} already present")
        return events
    seed_dir.mkdir(parents=True, exist_ok=True)
    out = seed_dir / "out"
    cfg = seed_dir / "config.xml"

    if args.matsim_runner == "org.matsim.run.RunMatsim":
        bs.build_matsim_config(
            Path(args.matsim_config), Path(args.network), Path(args.population),
            out, seed, args.threads, cfg,
            # Applied last, so they win over the benchmark's no-I/O settings.
            ["controller.writeEventsInterval=1", "controller.dumpDataAtEnd=true"]
            + list(args.matsim_param),
            args.capacity_factor, args.inject_activity_params,
        )
        cmd = ["java", f"-Xmx{args.heap_gb}g", "-Xms4g", "-cp", args.matsim_jar,
               args.matsim_runner, str(cfg)]
    else:
        # MATSimApplication-style runner: configured by --config: overrides, with
        # the two output ones flipped back on.
        cmd = ["java", f"-Xmx{args.heap_gb}g", "-Xms4g", "-cp", args.matsim_jar,
               args.matsim_runner, "run",
               "--config", str(Path(args.matsim_config).resolve()),
               "--iterations", "0", "--output", str(out), "--post-processing", "disabled"]
        shim = argparse.Namespace(
            population=args.population, network=args.network, seed=seed,
            capacity_factor=args.capacity_factor, matsim_param=list(args.matsim_param),
        )
        for kv in bs._app_overrides(shim, threads=args.threads):
            if kv.startswith(("controller.writeEventsInterval",
                              "controller.dumpDataAtEnd")):
                continue
            cmd += ["--config:" + kv]
        cmd += ["--config:controller.writeEventsInterval=1",
                "--config:controller.dumpDataAtEnd=true"]

    print(f"[seed {seed}] {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    # JAVA_TOOL_OPTIONS is set to -Xmx2g on this cluster's image and silently
    # overrides -Xmx, capping the heap at 2 GB whatever we ask for.
    env.pop("JAVA_TOOL_OPTIONS", None)
    log = seed_dir / "matsim.log"
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        print(f"ABORT: seed {seed}: MATSim exited {rc}; see {log}", file=sys.stderr)
        raise SystemExit(1)

    # MATSim names the events file after the run id when one is set, so find it
    # rather than assuming, then normalise to the name the validator looks for.
    found = sorted(out.rglob("*output_events.xml.gz")) or sorted(out.rglob("*events.xml.gz"))
    if not found:
        print(f"ABORT: seed {seed}: no events file under {out}", file=sys.stderr)
        raise SystemExit(1)
    if found[0] != events:
        shutil.move(str(found[0]), events)
    size_gb = events.stat().st_size / 1e9
    print(f"[seed {seed}] wrote {events} ({size_gb:.2f} GB)")

    # The validator reads per-trip travel times from output_trips.csv.gz, NOT from
    # the events, so it has to survive the cleanup below or the Wasserstein and KS
    # comparisons have nothing to work with on the MATSim side.
    trips = seed_dir / "output_trips.csv.gz"
    found_trips = sorted(out.rglob("*output_trips.csv.gz"))
    if found_trips:
        shutil.move(str(found_trips[0]), trips)
    else:
        print(f"WARNING: seed {seed}: no output_trips.csv.gz; travel-time "
              f"distribution comparison will be unavailable for this seed",
              file=sys.stderr)

    if args.keep_output_dir:
        return events
    shutil.rmtree(out, ignore_errors=True)
    return events


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matsim-jar", required=True)
    p.add_argument("--matsim-config", required=True)
    p.add_argument("--matsim-runner", default="org.matsim.run.RunMatsim")
    p.add_argument("--matsim-param", action="append", default=[])
    p.add_argument("--inject-activity-params", action="store_true")
    p.add_argument("--network", required=True)
    p.add_argument("--population", required=True)
    p.add_argument("--capacity-factor", type=float, required=True,
                   help="Must equal the TorchDNL side's --capacity-factor.")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--heap-gb", type=int, default=100)
    p.add_argument("--keep-output-dir", action="store_true",
                   help="Keep MATSim's full output tree, not just the events file.")
    p.add_argument("--force", action="store_true",
                   help="Re-run seeds whose events file already exists.")
    args = p.parse_args()

    bs = _load_bench_scaling()
    for seed in args.seeds:
        run_seed(args, bs, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
