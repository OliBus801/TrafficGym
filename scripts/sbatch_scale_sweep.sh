#!/bin/bash
# SLURM array job: one task = one cell of the controlled population-scale grid.
#
# Submit ONE ARRAY PER S LEVEL, because S is this experiment's cost axis and an
# array shares a single --time across all its tasks:
#
#   sbatch --time=2:00:00  --array=0-29%8  --export=ALL,TG_S=1  scripts/sbatch_scale_sweep.sh
#   sbatch --time=12:00:00 --array=0-59%8  --export=ALL,TG_S=16 scripts/sbatch_scale_sweep.sh
#
# --time must come from a measurement at THAT S level on THIS cluster. Our site's GPU
# buckets: b1<=3h, b2<=12h, b3<=24h, b4<=3d, b5<=7d — round down into a bucket.
#
#SBATCH --job-name=tg_scale
# The Slurm ACCOUNT is not baked in: pass it per submission
# (`sbatch --account=...`, or export SBATCH_ACCOUNT). Allocations differ per site and
# the right one also depends on which of yours has fairshare left.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/scale_%A_%a.out
#SBATCH --error=logs/scale_%A_%a.err

set -euo pipefail

REPO="${TG_REPO:?ABORT: set TG_REPO to the repository checkout}"
VENV="${TG_VENV:?ABORT: set TG_VENV to a Python environment with this package installed}"
SCENARIO=${TG_SCENARIO:-$REPO/tamarl/data/scenarios/sioux_falls}
OUTDIR=${TG_OUTDIR:-$REPO/results/scale_sweep}
EPISODES=${TG_EPISODES:-100}

# Either TG_S=<level> (production grid) or TG_PROBE=1 (loading probe).
GRID_SEL="--s ${TG_S:-}"
if [ -n "${TG_PROBE:-}" ]; then
  GRID_SEL="--probe ${TG_PROBE_SET:-load}"
elif [ -z "${TG_S:-}" ]; then
  echo "ABORT: set TG_S=<level> or TG_PROBE=1" >&2
  exit 1
fi

cd "$REPO"
# Must match the venv's interpreter (its pyvenv.cfg says 3.11.5), not a newer one.
module load python/3.11.5 >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- Resolve the cell BEFORE anything expensive ------------------------------
# Capture and test the exit status: `eval "$(cmd)"` would discard a refusal and
# leave the CELL_* variables unset, running the wrong thing under no tag.
if ! ASSIGN="$(python3 scripts/aamas_scale_grid.py $GRID_SEL --index "${SLURM_ARRAY_TASK_ID}")"; then
  echo "ABORT: grid mapper refused $GRID_SEL index=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
eval "$ASSIGN"
echo "[grid] $GRID_SEL task=${SLURM_ARRAY_TASK_ID} -> ${CELL_TAG}"

# --- Gates -------------------------------------------------------------------
echo "[gate] node=$(hostname) job=${SLURM_JOB_ID}"
NGPU=$(nvidia-smi -L | wc -l)
echo "[gate] visible GPUs: $NGPU"
if [ "$NGPU" -lt 1 ]; then echo "ABORT: no GPU allocated" >&2; exit 1; fi
python3 -c "import tamarl, torch; print('[gate] tamarl from', tamarl.__file__); print('[gate] torch', torch.__version__)"
python3 -c "
import sys, tamarl
want = '$REPO'
if not tamarl.__file__.startswith(want):
    sys.exit(f'ABORT: imported tamarl from {tamarl.__file__}, expected a path under {want}')
"

mkdir -p "$OUTDIR"

# --- Run ---------------------------------------------------------------------
srun python3 scripts/run_scale_cell.py \
  --scenario "$SCENARIO" \
  --population "$CELL_POP_TAG" \
  --method "$CELL_METHOD" \
  --seed "$CELL_SEED" \
  --scale_factor "$CELL_SCALE_FACTOR" \
  --arm "$CELL_ARM" \
  --s_level "$CELL_S" \
  --tag "$CELL_TAG" \
  --episodes "$EPISODES" \
  --device cuda \
  --output_dir "$OUTDIR"

echo "[ok] ${CELL_TAG}"
