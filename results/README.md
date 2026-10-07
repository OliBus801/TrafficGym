# Measured results

Every number in [RESULTS.md](../RESULTS.md) comes from a CSV in this directory. Nothing
here is extrapolated, smoothed or hand-edited.

`RESULTS.md` is generated, not written: run `./scripts/regenerate_results.sh` and
`git diff --exit-code RESULTS.md` should come back empty. That check needs no GPU, no
cluster account and no scenario data — it reads only the files committed here.

## How to read these files

One CSV per *cell*, where a cell is one (scenario, loading, engine, device, seed). The
file name is the cell: `berlin_L0p5_torchdnl_cpu_s2_bench.csv` is Berlin at loading 0.5,
TorchDNL on CPU, seed 2. `L0p5` means loading 0.5; `L32` means 32.

**`benchmarks/all_cells.csv` is the same data as one tidy table** — 459 rows, one per
cell, with two columns added that the per-cell files do not carry:

| column | meaning |
|---|---|
| `ladder` | which directory, i.e. which measurement protocol the cell belongs to |
| `device` | `gpu` or `cpu`. The `engine` column says `torchdnl` for both; only the tag distinguishes them |

Start there unless you want a specific cell's full instrumentation.

Timing columns: **`t_engine_s` is the number to compare.** It is the engine's own loop.
`t_total_s` includes scenario parsing and tensor construction, which differ by engine for
reasons that have nothing to do with simulation throughput. `t_setup_s` is the difference.

## Layout

```
benchmarks/
├── all_cells.csv                  every timing cell, tidy, all ladders
├── berlin/                        PRIMARY wall-clock ladder, car-only, post-fix engine
├── los_angeles/                   PRIMARY
├── sioux_falls/                   the small-network counterexample
└── superseded/
    ├── berlin_raw_prefix/         raw (not car-only) population, pre-fix engine
    ├── los_angeles_raw_prefix/    same
    ├── berlin_24h_fixed_capacity/         earliest protocol: 24 h horizon, capacity not scaled
    └── sioux_falls_24h_fixed_capacity/    same

validation/
├── fidelity_geh.csv               the GEH tables below, tidy
├── berlin_1pct_car/summary.md             PRIMARY, 20 seeds vs MATSim
├── los_angeles_1pct_car/summary.md        PRIMARY, 20 seeds vs MATSim
└── berlin_1pct_car_prefix_engine/summary.md   same data, pre-fix engine: what the defect cost

memory/
├── saturation/                    VRAM ceiling sweep, Berlin replicated 16x-347x
└── vram/                          single-GPU baseline footprint

learning/
├── calibration/                   episode-count and scale sizing for the training campaign
├── probe_scale/                   which capacity scale factors produce congestion
├── probe_convergence/             how many episodes UCB / MSA need
└── probe_top_k/                   candidate-route count sensitivity

ablations/
├── rng_seed_1/                    GPU-vs-CPU determinism check
└── rng_seed_2/
```

`learning/` and `ablations/` are sizing probes, not reported results. They are here
because they are what the campaign's parameters were chosen from.

## Which file backs which claim

| Claim | Files |
|---|---|
| Berlin wall-clock table, 4 engines x 9 loadings | `benchmarks/berlin/` |
| Los Angeles wall-clock table, 4 engines x 8 loadings | `benchmarks/los_angeles/` |
| GPU loses on a 76-link network | `benchmarks/sioux_falls/` |
| The superseded speed-ups, and the size of the correction | `benchmarks/superseded/` |
| DNL fidelity: GEH, envelope inclusion, Wasserstein | `validation/*/summary.md`, `validation/fidelity_geh.csv` |
| What the single-link-leg defect cost | `validation/berlin_1pct_car_prefix_engine/summary.md` |
| 37.6M concurrent agents on one 80 GB H100 | `memory/saturation/sat_berlin_x347_bench.csv` |
| Forced-advancement rate tracks the arrival rate | `fa_forced_move_rate` in `benchmarks/*/` |

## What is not here, and why

The per-seed DNL event dumps behind the fidelity tables are 2.6-3.0 GB per seed, 20 seeds
per scenario per engine — 50-60 GB per scenario — plus the MATSim `output_events.xml.gz`
they are compared against, at ~500 MB per seed. Those stay on the cluster.
`scripts/sbatch_validate_fidelity.sh` regenerates them and
`scripts/reduce_validation_events.py` reduces them to the per-seed `reduced.npz` that
`scripts/make_validation_figures.py` reads; the `summary.md` files here are that
pipeline's final output.

## Provenance and anonymity

The timing harness records which compute node and which Slurm job produced each cell.
Node names identify the cluster, so they cannot ship in a double-blind artifact.
`scripts/anonymize_results.py` is the only step between the cluster and this directory;
it is committed so you can see exactly what was changed:

- `hostname` -> `node-01` ... `node-59`, a stable mapping. "These two cells ran on the
  same node" survives; which cluster it was does not.
- `slurm_job` -> `job-001` ... `job-376`, same.
- `log` column dropped. It held an absolute path on a node-local disk that no longer
  exists.

Every other column is byte-for-byte what the harness wrote. The hardware itself is not
hidden: `cpu` is intact (AMD EPYC 9454, 48 cores) and the GPU is an 80 GB H100, visible
as `vram_total_gb` in `memory/`.
