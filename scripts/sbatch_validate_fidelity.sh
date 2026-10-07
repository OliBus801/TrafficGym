#!/bin/bash
# DNL-fidelity validation: replay one scenario's fixed car routes through BOTH
# engines across N seeds, so scripts/validate_against_matsim.py can compare them.
#
# One array task = one seed, or pass TG_SEEDS to walk several in one job.
#
# Scenario choice follows one rule: hold the trip count roughly constant and let the
# NETWORK grow, so the comparison isolates network size. The published appendix runs
# Ingolstadt (1,035 trips), a 32x32 grid (20,000) and Sioux Falls (131,516 trips on
# 3,968 links); Berlin's 10% sample adds 392,671 trips on 159,039 car links, and Los
# Angeles' 1% sample adds 317,803 trips on 1,154,727 -- 7.26x Berlin's network at
# comparable demand.
#
# The population MUST be car-only (scripts/filter_population_car_only.py). On the raw
# plans MATSim also teleports the walk/pt/bike legs, and both parsers mis-pair the
# zero-duration "car interaction" placeholders with car legs, so the two engines
# would be handed different demand.
#
# The ACCOUNT and any GPU request are passed per submission: only the TorchDNL stage
# needs a GPU. These runs write events, which the timing benchmark forbids, so they
# are NOT timings and are under no obligation to share a node type with anything.
#
#   sbatch --account=<alloc>_gpu --gpus=h100:1 --time=3:00:00 --mem=192G \
#          --export=ALL,TG_SCEN=los_angeles,TG_STAGE=torchdnl,TG_SEEDS='0 1 2' ...
#   sbatch --account=<alloc>_cpu --array=0-19 \
#          --export=ALL,TG_SCEN=los_angeles,TG_STAGE=matsim ...
#
#SBATCH --job-name=tg_valfid
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
# One node on our site had a broken GPU for the duration of this campaign; exclude
# any such node per submission with `sbatch --exclude=...`.
#SBATCH --output=logs/valfid_%A_%a.out
#SBATCH --error=logs/valfid_%A_%a.err

set -euo pipefail

REPO="${TG_REPO:?ABORT: set TG_REPO to the repository checkout}"
VENV="${TG_VENV:?ABORT: set TG_VENV to a Python environment with this package installed}"
MATSIM_ROOT="${TG_MATSIM_ROOT:?ABORT: set TG_MATSIM_ROOT to the MATSim install, jar plus scenario configs}"
LA_ROOT="${TG_LA_ROOT:?ABORT: set TG_LA_ROOT to the matsim-los-angeles checkout}"
MAX_STEPS=${TG_MAX_STEPS:-129600}        # 36 h, the horizon every paper experiment used
SEEDS=${TG_SEEDS:-${SLURM_ARRAY_TASK_ID:-0}}

: "${TG_SCEN:?ABORT: set TG_SCEN (berlin|los_angeles)}"

MATSIM_EXTRA=()
case "$TG_SCEN" in
  berlin)
    SCENDIR=$REPO/tamarl/data/scenarios/berlin
    NETWORK=$SCENDIR/berlin-v6.0-network-with-pt.xml
    POP_TAG=${TG_POP:-B1car}
    # The shipped sample is 10%, so capacity follows at 0.1 -- MATSim's own
    # downsampling convention, and it must be identical on both sides.
    CAP=${TG_CAPACITY:-0.1}
    MATSIM_CONFIG=$MATSIM_ROOT/scenarios/berlin/berlin-v6.0.config.xml
    MATSIM_RUNNER=org.matsim.project.OpenBerlinScenario
    ;;
  los_angeles)
    SCENDIR=$REPO/tamarl/data/scenarios/los_angeles
    NETWORK=$SCENDIR/la-v1.0-network.xml
    POP_TAG=${TG_POP:-L0p01car}
    CAP=${TG_CAPACITY:-0.01}
    LA_INPUT=$LA_ROOT/scenarios/los-angeles-v1.1/input
    MATSIM_CONFIG=$LA_INPUT/los-angeles-v1.1-1pct.config.xml
    # Its own runner is a pre-2023 main(args) that reads args[0] as the config path
    # and takes no --config: overrides, and using it would also measure a different
    # MATSim version from Berlin. Go through the same 2024.0 jar and RunMatsim.
    MATSIM_RUNNER=org.matsim.run.RunMatsim
    # Two LA-specific needs: the mode vehicle types (its config names them by an
    # absolute path that no longer exists), and activityParams generated from the
    # population, because LA encodes each activity's typical duration in its type
    # name (home_33000) and a static config cannot list them.
    MATSIM_EXTRA=(--matsim-param "vehicles.vehiclesFile=$LA_INPUT/los-angeles-v1.0-mode-vehicle-types_2019-12-05.xml"
                  --inject-activity-params)
    ;;
  *) echo "ABORT: unknown TG_SCEN $TG_SCEN" >&2; exit 1 ;;
esac

POP=$SCENDIR/${POP_TAG}_population.xml
VALROOT=${TG_VALROOT:-$REPO/results/validation_${TG_SCEN}_$POP_TAG}

: "${TG_STAGE:?ABORT: set TG_STAGE (torchdnl|matsim)}"

cd "$REPO"
module load java/17.0.6 >/dev/null 2>&1 || module load java >/dev/null 2>&1
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "[gate] node=$(hostname) scen=$TG_SCEN stage=$TG_STAGE seeds=$SEEDS capacity=$CAP pop=$POP_TAG"
test -f "$NETWORK" || { echo "ABORT: missing $NETWORK" >&2; exit 1; }
test -f "$POP"     || { echo "ABORT: missing $POP" >&2; exit 1; }

case "$TG_STAGE" in
  torchdnl)
    # TG_DEVICE lets the same stage run on CPU. The fidelity comparison is
    # device-independent -- it asks whether the QUEUE MODEL agrees with MATSim, not
    # how fast it runs -- so a CPU run is a valid appendix row and, on a saturated
    # cluster, reaches an answer sooner than queuing for an H100. The two devices do
    # not produce bit-identical seeds (the RNG stream differs), which is exactly why
    # agreement measured on either is a statement about the model.
    DEVICE=${TG_DEVICE:-cuda}
    if [ "$DEVICE" = "cuda" ]; then
      # A MIG slice is plenty: this scenario peaked at 0.26 GB of VRAM in the
      # benchmark, against 85 GB on a whole card. Asking for a fraction schedules
      # far sooner and stops holding a card we cannot fill.
      nvidia-smi -L || { echo "ABORT: no GPU allocated; submit with --gpus=... or set TG_DEVICE=cpu" >&2; exit 1; }
    fi
    srun python3 scripts/run_torchdnl_fixed_routes.py \
      --network "$NETWORK" --population "$POP" \
      --capacity-factor "$CAP" --max-steps "$MAX_STEPS" \
      --device "$DEVICE" --link-tt-interval 900 \
      --seeds $SEEDS --output-dir "$VALROOT/torchdnl"
    ;;
  matsim)
    srun python3 scripts/run_matsim_dnl_seeds.py \
      --matsim-jar "$MATSIM_ROOT/matsim-example-project-0.0.1-SNAPSHOT.jar" \
      --matsim-config "$MATSIM_CONFIG" \
      --matsim-runner "$MATSIM_RUNNER" "${MATSIM_EXTRA[@]}" \
      --network "$NETWORK" --population "$POP" \
      --capacity-factor "$CAP" --threads "${SLURM_CPUS_PER_TASK:-4}" \
      --heap-gb "${TG_HEAP_GB:-100}" \
      --seeds $SEEDS --output-dir "$VALROOT/matsim"
    ;;
  *) echo "ABORT: unknown TG_STAGE $TG_STAGE" >&2; exit 1 ;;
esac

echo "[ok] stage=$TG_STAGE seeds=$SEEDS"
