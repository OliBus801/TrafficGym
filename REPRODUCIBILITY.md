# Reproducing the Paper's Results

The experimental setup behind every reported number, and how to re-run it. See
[README.md](README.md) for installation, [DATA.md](DATA.md) for scenario provenance and
licensing, and [RESULTS.md](RESULTS.md) for the measured numbers — generated from the CSVs
under [results/](results/README.md).

The cheapest check needs no GPU, no cluster account and no scenario download:

```bash
./scripts/regenerate_results.sh && git diff --exit-code RESULTS.md
```

That rebuilds RESULTS.md from the committed CSVs. An empty diff means every number in it
traces to measured data in this repository.

## Hardware and software

Two campaigns ran on different machines.

**Wall-clock scaling and DNL fidelity** (benchmark tables, validation appendix):

- 1x NVIDIA H100 80 GB HBM3 per job, 2x AMD EPYC 9454 (48 cores) per node, 4 CPU cores per
  task, 64 GB job memory (Berlin, Sioux Falls), 128-192 GB (Los Angeles).
- Every engine in a comparison ran on the same node type with the same CPU allocation.
  MATSim and SUMO never touch the GPU; they request one anyway, so that every engine sees
  an identical CPU, memory system and bandwidth. The comparison is hardware-normalised.
- `TorchDNL` is single-process, no multi-GPU parallelism. Every cell fits in a 10 GB MIG
  slice; the heaviest, Berlin at loading 32, reserved 5.89 GB.
- CPU baselines: MATSim release 2024.0 and Eclipse SUMO 1.26.0, the latter in mesoscopic
  mode.

**Routing performance and entropy** (main results table):

- 4x NVIDIA A100-SXM4 (40 GB HBM2) node, 2x AMD EPYC 7413 (48 cores, 2.65 GHz), one GPU
  partition per run sized to the scenario: 1/7 compute / 5 GB for Sioux Falls, 3/7 / 20 GB
  for Berlin and Los Angeles. Job memory 64 GB (Sioux Falls, Berlin), 128 GB (Los
  Angeles), 8 CPU cores per task.

Sioux Falls-scale results need neither machine; a single consumer GPU is enough.

## Main results table

Routing performance and entropy across Sioux Falls, Berlin and Los Angeles, from
`tamarl.rl.train_bandit` via the JSON configs in `tamarl/data/configs/`, for each of the
nine evaluated agents (`random`, `epsilon_greedy`, `ucb`, `ts`, `exp3`, `msa`, `aon`,
`evo_swap`, `rd`), 5 seeds each, 100 episodes:

```bash
# Sioux Falls (ships with the repo, no download needed)
python -m tamarl.rl.train_bandit --config tamarl/data/configs/config_SF.json --agent ucb --seed 0

# Berlin / Los Angeles (fetch data first, see DATA.md)
python scripts/fetch_scenario_data.py --scenario berlin
python -m tamarl.rl.train_bandit --config tamarl/data/configs/config_Berlin.json --agent ts --seed 0
```

Two campaign values are not the committed defaults and must be passed explicitly:
`--max_steps 129600` (36 h; the configs say 115,200) and `--link_tt_interval 300`
(the `train()` default is 60 s, and MSA reads its update target off this binning).

```bash
python -m tamarl.rl.train_bandit --config tamarl/data/configs/config_Berlin.json \
  --agent ts --seed 1 --max_steps 129600 --link_tt_interval 300
```

Everything else matches: 100 episodes, `top_k_paths=9` (Link Penalty enumeration, penalty
factor 1.5x), `Delta t = 1.0 s`, stuck-agent forced-entry threshold 10 steps,
epsilon-compliance tolerance 60 s, seeds 1-5, mean +/- 1 std reported.

The two classical-assignment baselines, `msa` and `rd`, operate on OD-pair flows and take
`--formulation od_pair`; `scripts/sbatch_formulation_rerun.sh` runs them.

### Algorithm hyperparameters (framework defaults, unmodified)

| Algorithm | Hyperparameter | Value |
|---|---|---|
| epsilon-Greedy | epsilon start -> end | 1.0 -> 0.05 |
| | Multiplicative decay/episode | 0.995 |
| | EMA learning rate (alpha) | 0.1 |
| UCB | Exploration constant (c) | 100.0 |
| Thompson Sampling | Prior mean | -FFTT(p) per candidate path |
| | Prior std (sigma_0) | 10,000 (flat/uninformative) |
| | Environmental std (sigma_env) | max(10.0, std_p(FFTT(p))) per block |
| Exp3 | Learning rate (eta) | 0.01 |
| | Uniform mixing weight (gamma) | 0.05 |
| Replicator Dynamics | Inverse temperature (beta) | 0.1 |
| | Exploration noise start -> end | 0.05 -> 1e-8 |
| | Noise decay/episode | 0.99 |
| MSA | Step size start -> end | 1.0 -> 0.05 |
| | Decay | `alpha_n = alpha_min + (alpha_max - alpha_min) * exp(-0.01 n)` |

These are the CLI defaults in `tamarl/rl/train_bandit.py` and the JSON configs; every one
is overridable (`--ucb_c`, `--epsilon_start`, `--exp3_eta`, `--rd_beta`, `--ts_prior_std`,
...).

## DNL-vs-MATSim fidelity validation

`TorchDNL` is validated against MATSim's own DNL engine by replaying identical fixed routes
through both simulators' pure network-loading step, no replanning, which isolates
DNL-engine disagreement from route-choice disagreement.

Scenario choice holds the trip count roughly constant and lets the **network** grow:

| Scenario | Trips | Car links | Seeds |
|---|---|---|---|
| Ingolstadt | 1,035 | — | 10 |
| Grid 32x32 | 20,000 | — | 10 |
| Sioux Falls | 131,516 | 3,968 | 10 |
| Berlin, 10% sample | 392,671 | 159,039 | 20 |
| Los Angeles, 1% sample | 317,803 | 1,154,727 | 20 |

Reported metrics: the GEH statistic, MATSim inter-seed envelope inclusion, and the pooled
trip travel-time Wasserstein distance. Harness: `tamarl/validation/`,
`scripts/validate_against_matsim.py`, `scripts/make_validation_figures.py`.

**The population must be car-only — run `scripts/filter_population_car_only.py` first.**
On raw MATSim plans, MATSim also teleports the walk, pt, bike and ride legs, and both
parsers mis-pair the zero-duration `car interaction` placeholders with car legs, so the two
engines would be handed different demand.

Three stages, each a separate Slurm array:

```bash
# 1. replay the routes through each engine, one array task per seed
sbatch --account=<alloc>_gpu --gpus=h100:1 --time=3:00:00 --mem=192G \
  --export=ALL,TG_SCEN=berlin,TG_STAGE=torchdnl,TG_SEEDS='0 1 2' scripts/sbatch_validate_fidelity.sh
sbatch --account=<alloc>_cpu --array=0-19 \
  --export=ALL,TG_SCEN=berlin,TG_STAGE=matsim scripts/sbatch_validate_fidelity.sh

# 2. reduce each seed's event stream to per-bin series + link volumes
sbatch --account=<alloc>_cpu --array=0-19 \
  --export=ALL,TG_SCEN=berlin,TG_ENGINE=matsim scripts/sbatch_reduce_validation.sh

# 3. compare, and emit summary.md + the figures
python scripts/validate_against_matsim.py --val-root <validation dir>
python scripts/make_validation_figures.py --val-root <validation dir>
```

Stage 1 writes 2.6-3.0 GB of events per seed per engine — 50-60 GB per scenario — so only
stage 2's reduction and stage 3's `summary.md` are committed, under `results/validation/`.

## Wall-clock scaling benchmark

One Slurm array per engine, because the engines differ by two orders of magnitude in
walltime and an array shares a single `--time` across its tasks. `scripts/bench_grid.py`
maps an array index to a (loading, seed) cell:

```bash
# which index is which cell
python scripts/bench_grid.py --scenario berlin --engine torchdnl --index 0

# SUMO needs its network and routes converted first (netconvert, then per-loading routes)
NET=$(sbatch --parsable --export=ALL,TG_SCEN=berlin,TG_STAGE=net scripts/prep_sumo.sh)
sbatch --dependency=afterok:$NET --array=0-7 \
  --export=ALL,TG_SCEN=berlin,TG_STAGE=routes scripts/prep_sumo.sh

# one array per engine
sbatch --time=3:00:00  --array=0-29%10 \
  --export=ALL,TG_SCEN=berlin,TG_ENGINE=torchdnl scripts/sbatch_bench.sh
sbatch --time=12:00:00 --array=0-29%10 \
  --export=ALL,TG_SCEN=berlin,TG_ENGINE=matsim   scripts/sbatch_bench.sh
```

`TG_POP_SUFFIX` selects the population ladder and defaults to `car`, the car-only ladder
the paper quotes. `TG_POP_SUFFIX=""` reproduces the superseded raw-population numbers.

Figures:

```bash
PYTHONPATH=. python scripts/make_scaling_figure.py \
  --bench-dir results/benchmarks/berlin --network-label "Berlin, 159,039 car links" \
  --output figures/scaling_law
```

## Cluster scripts, and what they need

Nothing site-specific is baked into these scripts. Each one refuses to start until the
paths it needs are set, rather than silently running against the wrong tree:

| Variable | What it points at |
|---|---|
| `TG_REPO` | This repository's checkout, on a filesystem the compute nodes can read |
| `TG_VENV` | A Python environment with this package installed |
| `TG_MATSIM_ROOT` | MATSim install: the jar plus the per-scenario configs |
| `TG_LA_ROOT` | A `matsim-los-angeles` checkout (its configs are per sample size) |
| `TG_SUMO_BIN`, `TG_NETCONVERT` | The `sumo` and `netconvert` binaries |
| `TG_SUMO_PRELOAD` | `libatomic.so.1` on your site. The pip-installed SUMO binaries need it and many HPC images do not ship it on the default path. Preload exactly that one library: pointing `LD_LIBRARY_PATH` at a toolchain instead drags in its `libstdc++`, which then wants a newer glibc |

The Slurm **account** is never baked in either — pass `sbatch --account=...` or export
`SBATCH_ACCOUNT`. Log paths are relative (`logs/%A_%a.out`), resolved against the submit
directory, so `mkdir -p logs` before submitting.

| Script | What it runs |
|---|---|
| `scripts/sbatch_bench.sh` | One cell of the wall-clock ladder, any engine |
| `scripts/prep_sumo.sh` | SUMO network conversion, then per-loading route files |
| `scripts/sbatch_validate_fidelity.sh` | One seed of the fidelity replay, either engine |
| `scripts/sbatch_reduce_validation.sh` | Reduce one (engine, seed) event stream |
| `scripts/sbatch_scale_sweep.sh` | The controlled population-scale learning grid |
| `scripts/sbatch_formulation_rerun.sh` | MSA / RD in `od_pair`, Sioux Falls and Berlin |
| `scripts/sbatch_formulation_rerun_altcluster.sh` | The same, Los Angeles, on a second cluster |

Grid mappers (`scripts/bench_grid.py`, `scripts/aamas_scale_grid.py`) are the single source
of truth for which array index is which cell, and they refuse an out-of-range index rather
than defaulting.

## Collecting results off the cluster

`scripts/anonymize_results.py` is the only step between the cluster and `results/`. It
pseudonymizes the node and job columns the timing harness records and drops the node-local
log path; everything else is copied byte-for-byte. See the provenance section of
[results/README.md](results/README.md) for exactly what it changes.

```bash
rsync -a --include='*/' --include='*_bench.csv' --include='*_final.csv' \
  --include='*_curves.csv' --include='summary.md' --exclude='*' --prune-empty-dirs \
  <cluster>:<campaign>/results/ results/_raw/
python scripts/anonymize_results.py --raw results/_raw --out results
./scripts/regenerate_results.sh
```

## Compute scaling on synthetic grids

The GPU-vs-CPU throughput comparison on controlled network sizes uses the synthetic
`grid_world` family, generated at increasing `--size`/`--agents` with
`tamarl/data/scenarios/grid_world/generate_grid_scenario.py` and benchmarked with
`tests/benchmark_torchdnl.py` / `scripts/scale_matsim_network.py`.
