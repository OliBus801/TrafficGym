"""Wall-clock scaling benchmark: TorchDNL vs MATSim on the SAME dynamic network
loading, across a population-loading ladder.

Why this replaces the paper's headline speed-up figure
------------------------------------------------------
The submitted version quoted "up to 184x over state-of-the-art CPU simulators".
That number is against SUMO (39x against MATSim) on a 256x256 *synthetic grid* --
a regular lattice with uniform out-degree and contiguous link indexing, i.e. the
memory-access pattern most favourable to a GPU. Song et al. (2017) measured
exactly this trap: 10.0x on a grid collapsing to 2.37x on a real network. A
single maximum over baselines on the friendliest possible topology is not a
scaling law.

What this measures instead is the *shape* of the curve on a fixed real network as
loading grows. Losing to MATSim at low loading is an expected and reportable part
of that curve: TorchDNL advances a fixed 86,400 timesteps whatever the demand,
while MATSim's event-driven queue skips empty time, so a GPU kernel launch per
step dominates when the network is nearly empty. The claim worth making is about
how the two curves diverge, not about a single ratio.

Fairness protocol (all of it deliberate, all of it reportable)
--------------------------------------------------------------
1. SAME HARDWARE. Both engines run on identical `gpubase` nodes with the same
   --cpus-per-task. MATSim simply never touches the GPU on its node. This is the
   only configuration on this cluster where CPU model, memory and bandwidth are
   identical, and it answers the earlier review objection that the comparison was
   cross-hardware.
2. SAME WORK. One DNL of the same population over the same network, with routes
   taken verbatim from the population's <route> tags, so neither engine performs
   route choice and neither is credited with the other's routing decisions.
3. SAME QUEUE SEMANTICS. MATSim's stuckTime=10.0 s with removeStuckVehicles=false
   is exactly TorchDNL's stuck_threshold=10 at dt=1.0 s; flow and storage capacity
   factors are 1.0 on both sides; both step at 1 s.
4. NO OUTPUT ON EITHER SIDE. MATSim writes no events and no plans
   (writeEventsInterval=0, writePlansInterval=0) and TorchDNL runs with
   track_events=False and collect_link_tt=False. Disabling I/O favours MATSim,
   which is the direction an honest benchmark should lean.
5. TWO CLOCKS, BOTH REPORTED. `t_total_s` is the whole payload including
   interpreter/JVM start and input parsing -- what a user actually waits for.
   `t_engine_s` is the simulation alone. Reporting only the second would hide
   MATSim's XML parsing; reporting only the first would hide that a chunk of
   TorchDNL's cost is CUDA context creation.

Usage:
    python scripts/bench_scaling.py --engine torchdnl --network NET.xml \\
        --population POP.xml --seed 0 --tag sf_L1_s0 --output-dir DIR
    python scripts/bench_scaling.py --engine matsim --network NET.xml \\
        --population POP.xml --seed 0 --tag sf_L1_s0 --output-dir DIR \\
        --matsim-jar JAR --matsim-config TEMPLATE.xml --threads 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# 36 h, not 24 h: every experiment in the paper runs a 36-hour horizon so that
# trips departing late in the day can still finish. Both engines use it
# (MATSim via qsim.endTime below), so the comparison stays matched.
MAX_STEPS = 129600  # 36:00:00 at dt = 1 s


# --------------------------------------------------------------------------- #
# MATSim
# --------------------------------------------------------------------------- #
def _set_param(root: ET.Element, module: str, name: str, value: str) -> None:
    """Set <module name=...><param name=... value=.../></module>, creating as needed."""
    mod = None
    for m in root.findall("module"):
        if m.get("name") == module:
            mod = m
            break
    if mod is None:
        mod = ET.SubElement(root, "module", {"name": module})
    for p in mod.findall("param"):
        if p.get("name") == name:
            p.set("value", value)
            return
    ET.SubElement(mod, "param", {"name": name, "value": value})


def _controller_module(root: ET.Element) -> str:
    """MATSim renamed this module from `controler` to `controller`.

    Guessing wrong is silent and costly: _set_param would APPEND a second module
    under the wrong name, MATSim would keep honouring the original one, and every
    override below (output directory, iteration count, disabling event writing)
    would quietly do nothing.
    """
    names = {m.get("name") for m in root.findall("module")}
    for candidate in ("controller", "controler"):
        if candidate in names:
            return candidate
    return "controller"


def _network_crs(network: Path) -> str | None:
    """Read the CRS a MATSim network declares in its <attributes> header, if any.

    Only the first few KB are scanned: the attribute sits at the top of the file
    and these networks run to tens of megabytes.
    """
    import re

    with open(network, errors="replace") as f:
        head = f.read(16384)
    m = re.search(
        r'<attribute[^>]*name="coordinateReferenceSystem"[^>]*>([^<]+)</attribute>', head
    )
    return m.group(1).strip() if m else None


def _activity_types(population: Path) -> list[str]:
    """Every distinct activity type used by a population, in file order.

    Scanned with a line regex rather than a parser: these files run to gigabytes
    and only the `type` attribute of <activity> is wanted.
    """
    import re

    pat = re.compile(r'<activity[^>]*\stype="([^"]+)"')
    seen: dict[str, None] = {}
    with open(population, errors="replace") as f:
        for line in f:
            for m in pat.finditer(line):
                seen.setdefault(m.group(1), None)
    return list(seen)


def _inject_activity_params(root, population: Path) -> None:
    """Give the scoring module an entry for every activity type in the population.

    MATSim scores the plans once the mobsim of iteration 0 has finished, and throws
    `acttype "home_33000" is not known in utility parameters` for any type the
    config does not declare. LA encodes each activity's typical duration in the
    type name itself (`home_33000`, `eatout_11400`), which yields thousands of
    distinct types that its own runner generates programmatically -- so a static
    config can never list them.

    The entries are written with scoringThisActivityAtAll=false: this benchmark
    measures the mobsim, never the score, and switching scoring off per type keeps
    the run from inventing utility parameters that would look like a modelling
    choice. typicalDuration is still required by the parser, and is read back out
    of the type name where it is available.
    """
    mods = {m.get("name"): m for m in root.findall("module")}
    scoring = mods.get("scoring") or mods.get("planCalcScore")
    if scoring is None:
        scoring = ET.SubElement(root, "module")
        scoring.set("name", "scoring")

    declared = set()
    for ps in scoring.iter("parameterset"):
        if ps.get("type") == "activityParams":
            for prm in ps.findall("param"):
                if prm.get("name") == "activityType":
                    declared.add(prm.get("value"))

    added = 0
    for act in _activity_types(population):
        if act in declared:
            continue
        base, _, tail = act.rpartition("_")
        dur = tail if base and tail.isdigit() else "3600"
        ps = ET.SubElement(scoring, "parameterset")
        ps.set("type", "activityParams")
        for name, value in (("activityType", act),
                            ("typicalDuration", f"{int(dur) // 3600:02d}:"
                                                f"{int(dur) % 3600 // 60:02d}:"
                                                f"{int(dur) % 60:02d}"),
                            ("scoringThisActivityAtAll", "false")):
            prm = ET.SubElement(ps, "param")
            prm.set("name", name)
            prm.set("value", value)
        added += 1
    print(f"[matsim] activityParams: {len(declared)} already declared, {added} added")


def build_matsim_config(template: Path, network: Path, population: Path,
                        outdir: Path, seed: int, threads: int, dest: Path,
                        extra_params: list[str] | None = None,
                        capacity_factor: float = 1.0,
                        inject_activity_params: bool = False) -> None:
    tree = ET.parse(template)
    root = tree.getroot()

    # The template refers to its inputs RELATIVELY ("../Siouxfalls_facilities.xml.gz"),
    # which resolve against the config's own directory. The generated config lives
    # in $SLURM_TMPDIR, so every one of those paths must be rebased on the template
    # directory first -- not just the two we override below.
    tpl_dir = template.parent.resolve()
    for mod in root.findall("module"):
        for prm in mod.findall("param"):
            name, value = prm.get("name", ""), prm.get("value", "")
            if name.lower().endswith("file") and value and not os.path.isabs(value):
                prm.set("value", str((tpl_dir / value).resolve()))

    # Adopt the network's own coordinate reference system. Left unset, MATSim uses
    # its placeholder "Atlantis" CRS; that is fine for Sioux Falls, whose network
    # declares no projection, but Berlin's declares EPSG:25832 and MATSim then
    # tries to transform into a CRS no authority defines, failing at startup with
    # 'No authority was defined for code "ATLANTIS"'. Reading it from the network
    # keeps this correct for any scenario rather than hard-coding one city.
    crs = _network_crs(network)
    if crs:
        _set_param(root, "global", "coordinateSystem", crs)
        print(f"[matsim] coordinateSystem={crs} (from the network file)")

    ctl = _controller_module(root)
    _set_param(root, "network", "inputNetworkFile", str(network.resolve()))
    _set_param(root, "plans", "inputPlansFile", str(population.resolve()))
    _set_param(root, ctl, "outputDirectory", str(outdir.resolve()))
    _set_param(root, ctl, "firstIteration", "0")
    _set_param(root, ctl, "lastIteration", "0")
    _set_param(root, ctl, "overwriteFiles", "deleteDirectoryIfExists")
    # No I/O on either side: see fairness protocol point 4.
    _set_param(root, ctl, "writeEventsInterval", "0")
    _set_param(root, ctl, "writePlansInterval", "0")
    _set_param(root, ctl, "writeSnapshotsInterval", "0")
    _set_param(root, ctl, "dumpDataAtEnd", "false")
    # SAME WORK, continued. The baseline config has useTransit=true, so MATSim
    # would simulate pt and bus vehicles on top of car traffic while TorchDNL
    # simulates cars only. Leaving it on would charge MATSim for work we never do
    # and inflate our own speed-up.
    _set_param(root, "transit", "useTransit", "false")
    # useTransit=false is not enough on its own: MATSim still loads the transit
    # vehicles file named in the module ("loading transit vehicles from ..." in
    # the log). Drop the module so no transit input is read at all.
    for mod in list(root.findall("module")):
        if mod.get("name") in ("transit", "transitRouter"):
            root.remove(mod)
    # Counts are a calibration input; nothing in TorchDNL reads them and loading
    # them is pure overhead on MATSim's side. REMOVE the module rather than
    # blanking the path: MATSim resolves relative paths against the config's own
    # directory, so an empty inputCountsFile resolves back onto the config file
    # and MatsimCountsReader then chokes on it with a misleading
    # 'Doctype "config_v2.dtd" not known'.
    for mod in list(root.findall("module")):
        if mod.get("name") == "counts":
            root.remove(mod)
    # Queue semantics matched to TorchDNL: see point 3.
    _set_param(root, "qsim", "stuckTime", "10.0")
    _set_param(root, "qsim", "removeStuckVehicles", "false")
    _set_param(root, "qsim", "timeStepSize", "00:00:01")
    # Same simulated horizon on both sides, 36 h as in the paper's experiments.
    # Left unset, MATSim's qsim runs on until every agent has arrived, while
    # TorchDNL stops at MAX_STEPS -- under heavy loading that is a very different
    # amount of work, and the comparison would be meaningless.
    _set_param(root, "qsim", "startTime", "00:00:00")
    _set_param(root, "qsim", "endTime", "36:00:00")
    _set_param(root, "qsim", "flowCapacityFactor", f"{capacity_factor:g}")
    _set_param(root, "qsim", "storageCapacityFactor", f"{capacity_factor:g}")
    _set_param(root, "qsim", "numberOfThreads", str(threads))
    _set_param(root, "global", "numberOfThreads", str(threads))
    _set_param(root, "global", "randomSeed", str(seed))

    if inject_activity_params:
        _inject_activity_params(root, population)

    # Per-scenario overrides, applied last so they win. Needed because one
    # template cannot cover every scenario's conventions: Berlin's routes name the
    # vehicle they use (vehicleRefId="<person>_car"), which requires
    # qsim.vehiclesSource=modeVehicleTypesFromVehiclesData plus its mode-vehicle
    # types file, while Sioux Falls works with the default.
    for spec in extra_params or []:
        mod_name, _, value = spec.partition("=")
        module, _, name = mod_name.partition(".")
        if not (module and name):
            raise SystemExit(f"ABORT: bad --matsim-param {spec!r}; expected module.name=value")
        _set_param(root, module, name, value)
        print(f"[matsim] {module}.{name}={value}")
    # MATSim's config parser requires the DOCTYPE and ElementTree drops it, which
    # fails with a bare "this.delegate is null" NPE on the first tag. The template
    # points at a RELATIVE dtd path that would not resolve from $SLURM_TMPDIR
    # either, so emit the canonical URL: MATSim resolves matsim.org DTDs from its
    # own classpath and never touches the network.
    body = ET.tostring(root, encoding="unicode")
    dest.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">\n'
        f"{body}\n",
        encoding="utf-8",
    )


def _app_overrides(args, threads: int) -> list[str]:
    """The fairness protocol, expressed as MATSimApplication --config: overrides."""
    return [
        f"plans.inputPlansFile={Path(args.population).resolve()}",
        f"network.inputNetworkFile={Path(args.network).resolve()}",
        "controller.writeEventsInterval=0",
        "controller.writePlansInterval=0",
        "controller.dumpDataAtEnd=false",
        "qsim.stuckTime=10.0",
        "qsim.removeStuckVehicles=false",
        "qsim.timeStepSize=00:00:01",
        "qsim.startTime=00:00:00",
        "qsim.endTime=36:00:00",
        f"qsim.flowCapacityFactor={args.capacity_factor:g}",
        f"qsim.storageCapacityFactor={args.capacity_factor:g}",
        f"qsim.numberOfThreads={threads}",
        f"global.numberOfThreads={threads}",
        f"global.randomSeed={args.seed}",
        "transit.useTransit=false",
    ] + list(args.matsim_param)


def run_matsim(args, workdir: Path) -> dict:
    cfg = workdir / "config.xml"
    outdir = workdir / "out"
    if args.matsim_runner == "org.matsim.run.RunMatsim":
        build_matsim_config(
            Path(args.matsim_config), Path(args.network), Path(args.population),
            outdir, args.seed, args.threads, cfg, args.matsim_param,
            args.capacity_factor, args.matsim_inject_activity_params,
        )
    java = ["java", f"-Xmx{args.heap_gb}g", f"-Xms{min(4, args.heap_gb)}g",
            "-cp", args.matsim_jar]
    if args.matsim_runner == "org.matsim.run.RunMatsim":
        cmd = java + [args.matsim_runner, str(cfg)]
    else:
        # MATSimApplication runner (e.g. org.matsim.project.OpenBerlinScenario).
        # These carry the scenario's own prepareConfig()/prepareControler(), which
        # is where Berlin sets its CRS, vehicle source and mode vehicle types --
        # exactly the settings a bare RunMatsim leaves unset and that we were
        # otherwise patching by hand, one failure at a time. Use the scenario's own
        # config unmodified and express every override through --config: so the
        # application, not us, decides the defaults.
        cmd = java + [args.matsim_runner, "run",
                      "--config", str(Path(args.matsim_config).resolve()),
                      "--iterations", "0",
                      "--output", str(outdir.resolve()),
                      # Post-processing runs analyses after the mobsim; timing it
                      # would charge MATSim for work TorchDNL never does.
                      "--post-processing", "disabled"]
        for kv in _app_overrides(args, threads=args.threads):
            cmd += ["--config:" + kv]
    print("[cmd]", " ".join(cmd), flush=True)
    log_path = workdir / "matsim.log"
    # The site sets JAVA_TOOL_OPTIONS=-Xmx2g. An explicit -Xmx on the command line
    # wins, but a 2 GB default silently capping a 500k-agent run is not a risk
    # worth carrying, so drop it rather than relying on precedence.
    env = {k: v for k, v in os.environ.items() if k != "JAVA_TOOL_OPTIONS"}
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=workdir, env=env)
    t_total = time.perf_counter() - t0

    if proc.returncode != 0:
        tail = "".join(open(log_path).readlines()[-40:])
        print(f"ABORT: MATSim exited {proc.returncode}\n{tail}", file=sys.stderr)
        raise SystemExit(1)

    # Engine time = the span between MATSim's own iteration markers, read off its
    # log timestamps. An earlier version scanned those lines for the first
    # float-looking token and dutifully returned 0.0 -- the "0" in "ITERATION 0".
    # NaN is the right answer when the markers are absent; a wrong number that
    # looks plausible is worse than a missing one.
    t_engine = float("nan")
    t_begin = t_end = None
    for line in open(log_path, errors="replace"):
        marker = "ITERATION 0 BEGINS" if "ITERATION 0 BEGINS" in line else (
            "ITERATION 0 ENDS" if "ITERATION 0 ENDS" in line else None
        )
        if marker is None:
            continue
        stamp = line.split()[0]
        try:
            t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S,%f")
        except ValueError:
            continue
        if marker.endswith("BEGINS"):
            t_begin = t
        else:
            t_end = t
    if t_begin and t_end:
        t_engine = (t_end - t_begin).total_seconds()
    return {"t_total_s": t_total, "t_engine_s": t_engine, "log": str(log_path)}


# --------------------------------------------------------------------------- #
# SUMO
# --------------------------------------------------------------------------- #
def run_sumo(args, workdir: Path) -> dict:
    """Time one mesoscopic SUMO run over the same trips as the other engines.

    Mesoscopic mode (--mesosim) is the comparable model: SUMO's default is
    microscopic car-following, which solves a different problem at a different
    cost and would make the comparison meaningless.

    Output is disabled exactly as for MATSim and TorchDNL, and the horizon is the
    same MAX_STEPS seconds. SUMO's own step log is off so the run is not charged
    for console I/O.
    """
    for f in (args.sumo_net, args.sumo_routes):
        if not f or not Path(f).is_file():
            print(f"ABORT: missing SUMO input {f!r}", file=sys.stderr)
            raise SystemExit(1)
    cmd = [
        args.sumo_binary, "--mesosim",
        "--net-file", str(Path(args.sumo_net).resolve()),
        "--route-files", str(Path(args.sumo_routes).resolve()),
        "--begin", "0", "--end", str(MAX_STEPS),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--xml-validation", "never",
        "--time-to-teleport", "-1",       # never teleport: a stuck vehicle stays
        # netconvert refuses a small residue of U-turns that MATSim's queue model
        # allows and the demand uses, even when a connection file declares them
        # explicitly. Without this, the FIRST such vehicle aborts the whole run.
        # scripts/prep_sumo.sh measures how much demand this costs and caps it;
        # the count is written to <scenario>/sumo/route-connectivity.txt.
        "--ignore-route-errors", "true",
        "--seed", str(args.seed),
    ]
    print("[cmd]", " ".join(cmd), flush=True)
    log_path = workdir / "sumo.log"
    env = dict(os.environ)
    if args.sumo_preload:
        # The pip-installed SUMO binaries need libatomic.so.1, which the cluster's
        # base image does not carry. Preloading just that one library avoids
        # dragging in a whole toolchain's libstdc++ and its newer glibc dependency.
        env["LD_PRELOAD"] = args.sumo_preload
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=workdir, env=env)
    t_total = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "".join(open(log_path).readlines()[-30:])
        print(f"ABORT: SUMO exited {proc.returncode}\n{tail}", file=sys.stderr)
        raise SystemExit(1)
    return {"t_total_s": t_total, "t_engine_s": t_total, "log": str(log_path)}


# --------------------------------------------------------------------------- #
# TorchDNL
# --------------------------------------------------------------------------- #
def run_torchdnl(args) -> dict:
    t0 = time.perf_counter()
    import torch

    from tamarl.core.torchdnl import TorchDNL
    from tamarl.envs.scenario_loader import parse_network
    from tamarl.validation.matsim_io import parse_fixed_route_population

    _, edges_data, link_id_to_idx, _ = parse_network(
        args.network, args.capacity_factor, 1.0
    )
    edge_static = torch.tensor([e["attr"] for e in edges_data], dtype=torch.float32)
    edge_endpoints = torch.tensor([[e["u"], e["v"]] for e in edges_data], dtype=torch.int32)
    fr = parse_fixed_route_population(args.population, link_id_to_idx)
    departure_times = torch.tensor(fr.departure_times, dtype=torch.int32)
    act_end_times = torch.tensor(fr.act_end_times, dtype=torch.int32)
    act_durations = torch.tensor(fr.act_durations, dtype=torch.int32)
    num_legs = torch.tensor(fr.num_legs, dtype=torch.int32)
    paths_flat = torch.tensor(fr.paths_flat, dtype=torch.int32)
    path_offsets = torch.tensor(fr.path_offsets, dtype=torch.long)
    # In-memory replication for the saturation ladder. Filling an 80 GB H100 needs
    # ~350x the Berlin 10% sample; as XML that is ~300 GB on disk and hours of
    # parsing, so the file format, not the GPU, becomes the wall. Tiling the
    # already-parsed CSR arrays produces exactly the population a 350x XML clone
    # would have produced, in seconds. MATSim cannot be fed this way, which is why
    # the two-engine comparison uses the XML ladder and this path is TorchDNL-only.
    if args.replicate > 1:
        k = args.replicate
        n0 = len(fr.departure_times)
        span = int(path_offsets[-1])
        paths_flat = paths_flat.repeat(k)
        # Each copy's slice of paths_flat starts `span` further along than the last.
        base_off = path_offsets[:-1]
        path_offsets = torch.cat(
            [base_off + i * span for i in range(k)] + [torch.tensor([span * k], dtype=torch.long)]
        )
        departure_times = departure_times.repeat(k)
        act_end_times = act_end_times.repeat(k, 1)
        act_durations = act_durations.repeat(k, 1)
        num_legs = num_legs.repeat(k)
        n_agents = n0 * k
        print(f"[replicate] x{k}: {n0:,} -> {n_agents:,} agents, "
              f"{paths_flat.numel():,} path entries")
    else:
        n_agents = len(fr.departure_times)
    t_setup = time.perf_counter() - t0

    # Peak VRAM is a first-class result here, not a diagnostic: the point of the
    # Berlin ladder is to push the population until the GPU is saturated, and the
    # largest population that fits is what defines the top of that ladder.
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Instrumentation OFF: this is a timing run, and MATSim writes nothing either.
    t1 = time.perf_counter()
    dnl = TorchDNL(
        edge_static=edge_static,
        paths=None,
        device=args.device,
        departure_times=departure_times,
        edge_endpoints=edge_endpoints,
        act_end_times=act_end_times,
        act_durations=act_durations,
        num_legs=num_legs,
        dt=1.0,
        seed=args.seed,
        track_events=False,
        collect_link_tt=False,
        paths_flat=paths_flat,
        path_offsets=path_offsets,
        max_steps=MAX_STEPS,
    )
    while dnl.current_step < MAX_STEPS:
        if dnl.active_agents_count == 0:
            break
        dnl.step()
    if args.device == "cuda":
        torch.cuda.synchronize()
    dnl.finalize_stuck_agents()
    t_engine = time.perf_counter() - t1

    m = dnl.get_metrics()
    vram = {}
    if args.device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        vram = {
            "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "vram_total_gb": total_b / 1e9,
        }
    return {
        "t_total_s": t_setup + t_engine,
        "t_engine_s": t_engine,
        "t_setup_s": t_setup,
        "n_agents": n_agents,
        "n_path_entries": int(paths_flat.numel()),
        "replicate": args.replicate,
        **vram,
        "arrived": m["arrived_count"],
        "steps_run": dnl.current_step,
        **{k: v for k, v in m.items() if k.startswith("fa_")},
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True,
                   choices=("torchdnl", "matsim", "sumo"))
    p.add_argument("--network", required=True)
    p.add_argument("--population", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scenario", default="")
    p.add_argument("--loading", type=float, default=float("nan"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--capacity-factor",
        type=float,
        default=1.0,
        help="Flow AND storage capacity multiplier, applied identically to both "
        "engines (TorchDNL via parse_network's scale_factor, MATSim via "
        "qsim.flow/storageCapacityFactor). Scale it WITH the population: a p%% "
        "population sample on a 100%% network is nearly empty, so varying demand "
        "alone measures free flow rather than load. Holding V/C fixed instead "
        "makes the number of events to process the only variable.",
    )
    p.add_argument("--heap-gb", type=int, default=48)
    p.add_argument("--replicate", type=int, default=1,
                   help="TorchDNL only: tile the parsed population K times in memory "
                        "(saturation ladder; MATSim cannot be fed this way)")
    p.add_argument("--matsim-jar", default="")
    p.add_argument("--matsim-config", default="")
    p.add_argument("--sumo-binary", default="sumo")
    p.add_argument("--sumo-net", default="")
    p.add_argument("--sumo-routes", default="")
    p.add_argument("--sumo-preload", default="",
                   help="Path to libatomic.so.1 for LD_PRELOAD")
    p.add_argument("--matsim-runner", default="org.matsim.run.RunMatsim",
                   help="Java entry point; a MATSimApplication subclass such as "
                        "org.matsim.project.OpenBerlinScenario is invoked in app mode")
    p.add_argument("--matsim-inject-activity-params", action="store_true",
                   help="Declare an activityParams entry (scoring off) for every "
                        "activity type found in the population. Needed by Los "
                        "Angeles, whose type names carry their typical duration "
                        "(home_33000) and so cannot be listed in a static config.")
    p.add_argument("--matsim-param", action="append", default=[],
                   metavar="MODULE.NAME=VALUE",
                   help="Override a MATSim config param; repeatable")
    p.add_argument("--workdir", default="")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    done = outdir / f"{args.tag}.DONE"
    if done.exists():
        print(f"[skip] {args.tag} already complete")
        return 0
    outdir.mkdir(parents=True, exist_ok=True)

    for f in (args.network, args.population):
        if not Path(f).is_file():
            print(f"ABORT: missing input {f}", file=sys.stderr)
            return 1
    if args.engine == "sumo" and not (args.sumo_net and args.sumo_routes):
        print("ABORT: --sumo-net and --sumo-routes are required for sumo", file=sys.stderr)
        return 1
    if args.engine == "matsim" and not (args.matsim_jar and args.matsim_config):
        print("ABORT: --matsim-jar and --matsim-config are required for matsim", file=sys.stderr)
        return 1

    # Record the host so a later reader can verify the same-hardware claim
    # instead of taking it on trust.
    host = {"hostname": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    try:
        host["cpu"] = next(
            ln.split(":", 1)[1].strip()
            for ln in open("/proc/cpuinfo")
            if ln.startswith("model name")
        )
    except Exception:
        host["cpu"] = "unknown"

    workdir = Path(args.workdir or os.environ.get("SLURM_TMPDIR", "/tmp")) / args.tag
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        if args.engine == "matsim":
            res = run_matsim(args, workdir)
        elif args.engine == "sumo":
            res = run_sumo(args, workdir)
        else:
            res = run_torchdnl(args)
    finally:
        for engine_name in ("matsim", "sumo"):
            if args.engine == engine_name:
                log = workdir / f"{engine_name}.log"
                if log.exists():
                    shutil.copy(log, outdir / f"{args.tag}_{engine_name}.log")
        shutil.rmtree(workdir / "out", ignore_errors=True)

    row = {
        "tag": args.tag,
        "engine": args.engine,
        "scenario": args.scenario,
        "loading": args.loading,
        "seed": args.seed,
        "threads": args.threads,
        # Recorded so a reader can verify V/C was held constant across the
        # ladder rather than take it on trust.
        "capacity_factor": args.capacity_factor,
        "population_file": Path(args.population).name,
        **host,
        **res,
    }
    with open(outdir / f"{args.tag}_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    done.write_text(json.dumps(row, indent=2, default=str))
    print(f"[done] {args.tag} engine={args.engine} total={res['t_total_s']:.1f}s "
          f"engine={res.get('t_engine_s', float('nan')):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
