#!/bin/bash
# SLURM array job for the TorchDNL-vs-MATSim wall-clock scaling benchmark.
#
# BOTH engines run on gpubase nodes with the same --cpus-per-task. MATSim never
# touches the GPU on its node; requesting one anyway is the point, because it is
# the only way to guarantee the two engines see an identical CPU, memory system
# and bandwidth. That is what makes the comparison hardware-normalised rather
# than the cross-hardware measurement an earlier round of reviews objected to.
#
# Submit one array PER ENGINE (see scripts/bench_grid.py for why):
#   sbatch --time=3:00:00  --array=0-29%10 --export=ALL,TG_SCEN=sioux_falls,TG_ENGINE=torchdnl scripts/sbatch_bench.sh
#   sbatch --time=12:00:00 --array=0-29%10 --export=ALL,TG_SCEN=sioux_falls,TG_ENGINE=matsim   scripts/sbatch_bench.sh
#
#SBATCH --job-name=tg_bench
# A second GPU allocation rather than the campaign's main one: the main one's fairshare is spent
# (LevelFS 0.151, 82.7% of its share consumed by this campaign's own GPU cells),
# while the second is effectively untouched (RawUsage 2134, LevelFS 68106) and
# carries a 2.7x larger normalised share. Moving a pending cell across raised its
# priority from 1,258,599 to 2,028,056. Override per submission with
# `sbatch --account=...` if the allocation has to go back.
# The Slurm ACCOUNT is not baked in: pass it per submission
# (`sbatch --account=...`, or export SBATCH_ACCOUNT). Allocations differ per site and
# the right one also depends on which of yours has fairshare left.
# The GPU request is passed PER SUBMISSION, not fixed here, because every cell of
# this campaign fits in the smallest MIG slice: the heaviest is Berlin at loading 32
# with 5.89 GB reserved, against 10 GB in a 1g.10gb slice. A slice schedules 8 per
# node instead of 1, which on a saturated cluster is the difference between running
# and waiting, and it stops holding a whole card we cannot fill.
#
#   --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#
# The CPU-only engines (matsim, sumo, torchdnl_cpu) ask for a slice too. They never
# touch it; the point is that they land on the same gpubase node type as the GPU
# cells, which is what makes the comparison hardware-normalised rather than the
# cross-hardware measurement an earlier round of reviews objected to.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# One node on our site had a broken GPU for the duration of this campaign; exclude
# any such node per submission with `sbatch --exclude=...`.
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err

set -euo pipefail

REPO="${TG_REPO:?ABORT: set TG_REPO to the repository checkout}"
VENV="${TG_VENV:?ABORT: set TG_VENV to a Python environment with this package installed}"
OUTDIR=${TG_OUTDIR:-$REPO/results/bench}
THREADS=${SLURM_CPUS_PER_TASK:-4}
MATSIM_ROOT="${TG_MATSIM_ROOT:?ABORT: set TG_MATSIM_ROOT to the MATSim install, jar plus scenario configs}"
LA_ROOT="${TG_LA_ROOT:?ABORT: set TG_LA_ROOT to the matsim-los-angeles checkout}"
SUMO_BIN="${TG_SUMO_BIN:?ABORT: set TG_SUMO_BIN to the sumo binary}"
# The pip-installed SUMO binaries need libatomic.so.1, absent from the cluster
# image. Preload exactly that one library: pointing LD_LIBRARY_PATH at a
# toolchain instead drags in its libstdc++, which then wants a newer glibc.
SUMO_PRELOAD="${TG_SUMO_PRELOAD:?ABORT: set TG_SUMO_PRELOAD to libatomic.so.1 on this site}"
# TG_POP_SUFFIX picks which population ladder a cell runs on. It defaults to "car"
# -- the car-only ladder built by scripts/filter_population_car_only.py -- because
# the raw ladder is NOT a valid input for this comparison: it hands MATSim 2.2x more
# legs than TorchDNL simulates, and it makes both parsers pair the zero-duration
# "car interaction" placeholder to a car leg instead of the real activity, so
# multi-leg agents depart their later trips hours early. Set TG_POP_SUFFIX="" only
# to reproduce the pre-2026-10-02 numbers for the purpose of quantifying that bias.
POP_SUFFIX=${TG_POP_SUFFIX-car}
MATSIM_EXTRA=()
JAR=$MATSIM_ROOT/matsim-example-project-0.0.1-SNAPSHOT.jar

: "${TG_SCEN:?ABORT: set TG_SCEN}"
: "${TG_ENGINE:?ABORT: set TG_ENGINE}"

case "$TG_SCEN" in
  sioux_falls)
    SCENDIR=$REPO/tamarl/data/scenarios/sioux_falls
    NETWORK=$SCENDIR/Siouxfalls_network_PT.xml
    CFG=$MATSIM_ROOT/scenarios/siouxfalls-2014/configs/config_baseline_no_replan.xml
    ;;
  berlin)
    SCENDIR=$REPO/tamarl/data/scenarios/berlin
    NETWORK=$SCENDIR/berlin-v6.0-network-with-pt.xml
    # Berlin ships its own runner and its own config. OpenBerlinScenario's
    # prepareConfig() sets the CRS, the vehicle source and the mode vehicle types
    # that a bare RunMatsim leaves unset -- every one of which we had otherwise hit
    # as a separate startup failure. Use them instead of a hand-patched Sioux Falls
    # template.
    CFG=$MATSIM_ROOT/scenarios/berlin/berlin-v6.0.config.xml
    MATSIM_EXTRA=(--matsim-runner "org.matsim.project.OpenBerlinScenario")
    ;;
  los_angeles)
    # LA keeps its own config per sample size (whose capacity factors already equal
    # that fraction) but NOT its own jar. RunLosAngelesScenario is a plain
    # main(args) on a pre-2023 MATSim: it reads args[0] as the config path, so the
    # MATSimApplication "run" subcommand lands on it as a filename
    # (FileNotFoundException: run) and it accepts no --config: overrides at all.
    # Running LA on that jar would also measure a different MATSim version from
    # Berlin and Sioux Falls, making the three series incomparable. So LA goes
    # through the same MATSim 2024.0 jar and the same generic RunMatsim path as
    # Sioux Falls, with its own config as the template.
    SCENDIR=$REPO/tamarl/data/scenarios/los_angeles
    NETWORK=$SCENDIR/la-v1.0-network.xml
    LA_INPUT=$LA_ROOT/scenarios/los-angeles-v1.1/input
    # LA's config names its inputs by ABSOLUTE paths on the machine it was authored
    # on (an absolute path under the author's home on another machine), which no longer exist.
    # build_matsim_config only rebases RELATIVE paths, so the one input we still
    # need -- the mode vehicle types, required by vehiclesSource=
    # modeVehicleTypesFromVehiclesData -- must be pointed at explicitly. The dead
    # transit and counts paths are dropped with their modules.
    # LA names each activity after its typical duration (home_33000, eatout_11400),
    # so the config cannot list the types and MATSim throws in the scoring pass that
    # follows the mobsim. Its own runner generates those entries programmatically;
    # we generate them from the population instead.
    MATSIM_EXTRA=(--matsim-param "vehicles.vehiclesFile=$LA_INPUT/los-angeles-v1.0-mode-vehicle-types_2019-12-05.xml"
                  --matsim-inject-activity-params)
    ;;
  *) echo "ABORT: unknown scenario $TG_SCEN" >&2; exit 1 ;;
esac

cd "$REPO"
module load java/17.0.6 >/dev/null 2>&1 || module load java >/dev/null 2>&1
module load python/3.11.5 >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# TG_CELLS runs several grid indices in ONE job, in sequence. Preferred over an
# array of one cell each: a cell is minutes of compute, so an array of them queues
# dozens of times over on a busy cluster while one job walks them on a single
# allocation. Grouping a loading's three seeds also means a failure identifies the
# loading immediately.
CELLS=${TG_CELLS:-${SLURM_ARRAY_TASK_ID:-0}}

for CELL in $CELLS; do
  if ! ASSIGN="$(python3 scripts/bench_grid.py --scenario "$TG_SCEN" --engine "$TG_ENGINE" \
                 --index "$CELL")"; then
    echo "ABORT: grid mapper refused index=$CELL" >&2
    exit 1
  fi
  eval "$ASSIGN"
  echo "[grid] cell=$CELL -> ${CELL_TAG} pop=${CELL_POP_TAG}${POP_SUFFIX}"

  # LA's config is per sample size, so it can only be chosen once the cell is known.
  if [ "$TG_SCEN" = "los_angeles" ]; then
    case "$CELL_LOADING" in
      0.001) LA_PCT=0.1pct ;;
      # The two bracket-tightening loadings are subsamples of the 5% population, so
      # they borrow its config as a template. Its capacity factors are overridden
      # from CELL_CAPACITY_FACTOR regardless, and the activity params come from the
      # population itself.
      0.02)  LA_PCT=5pct   ;;
      0.03)  LA_PCT=5pct   ;;
      0.01)  LA_PCT=1pct   ;;
      0.05)  LA_PCT=5pct   ;;
      0.1)   LA_PCT=10pct  ;;
      0.25)  LA_PCT=25pct  ;;
      1.0)   LA_PCT=25pct ;;   # full city, cloned x4 from the 25% sample
      *) echo "ABORT: no LA config for loading $CELL_LOADING" >&2; exit 1 ;;
    esac
    CFG=$LA_ROOT/scenarios/los-angeles-v1.1/input/los-angeles-v1.1-$LA_PCT.config.xml
  fi

  # --- Gates, cheapest first ---------------------------------------------------
  echo "[gate] node=$(hostname) job=${SLURM_JOB_ID} threads=$THREADS"
  echo "[gate] cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  nvidia-smi -L || { echo "ABORT: no GPU allocated" >&2; exit 1; }

  POP=$SCENDIR/${CELL_POP_TAG}${POP_SUFFIX}_population.xml
  test -f "$NETWORK" || { echo "ABORT: missing network $NETWORK" >&2; exit 1; }
  test -f "$POP"     || { echo "ABORT: missing population $POP" >&2; exit 1; }

  EXTRA=()
  DEVICE=cuda
  case "$CELL_ENGINE" in
    matsim)
      test -f "$JAR" || { echo "ABORT: missing MATSim jar $JAR" >&2; exit 1; }
      test -f "$CFG" || { echo "ABORT: missing MATSim config $CFG" >&2; exit 1; }
      java -version 2>&1 | head -1
      # Leave headroom under --mem for the JVM's own off-heap use.
      EXTRA=(--matsim-jar "$JAR" --matsim-config "$CFG" --heap-gb "${TG_HEAP_GB:-160}" "${MATSIM_EXTRA[@]}")
      ;;
    sumo)
      SUMO_NET=$SCENDIR/sumo/network.net.xml
      SUMO_ROU=$SCENDIR/sumo/${CELL_POP_TAG}${POP_SUFFIX}.rou.xml
      test -f "$SUMO_NET" || { echo "ABORT: missing $SUMO_NET; run prep_sumo first" >&2; exit 1; }
      test -f "$SUMO_ROU" || { echo "ABORT: missing $SUMO_ROU; run prep_sumo first" >&2; exit 1; }
      EXTRA=(--sumo-binary "$SUMO_BIN" --sumo-net "$SUMO_NET" --sumo-routes "$SUMO_ROU"
             --sumo-preload "$SUMO_PRELOAD")
      ;;
    torchdnl_cpu)
      # Same engine, CPU device. It still runs on a gpubase node so that every
      # series in the figure was measured on identical hardware.
      DEVICE=cpu
      ;;
  esac

  ENGINE_ARG=$CELL_ENGINE
  [ "$CELL_ENGINE" = "torchdnl_cpu" ] && ENGINE_ARG=torchdnl

  srun python3 scripts/bench_scaling.py \
    --engine "$ENGINE_ARG" \
    --network "$NETWORK" \
    --population "$POP" \
    --seed "$CELL_SEED" \
    --tag "$CELL_TAG" \
    --scenario "$CELL_SCENARIO" \
    --loading "$CELL_LOADING" \
    --capacity-factor "$CELL_CAPACITY_FACTOR" \
    --threads "$THREADS" \
    --device "$DEVICE" \
    --output-dir "$OUTDIR" \
    "${EXTRA[@]}"

  echo "[ok] ${CELL_TAG}"
done

echo "[done] cells=$CELLS"
