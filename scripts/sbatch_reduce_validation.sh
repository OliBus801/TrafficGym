#!/bin/bash
# Reduce one scenario's validation seeds to per-bin series and link volumes.
# One array task = one (engine, seed). CPU-only and I/O-bound: no GPU, and the
# CPU pool is both wider and cheaper than the GPU allocation this campaign has
# already spent.
#
#   sbatch --account=<alloc>_cpu --array=0-19 \
#          --export=ALL,TG_SCEN=los_angeles,TG_ENGINE=matsim   scripts/sbatch_reduce_validation.sh
#   sbatch --account=<alloc>_cpu --array=0-19 \
#          --export=ALL,TG_SCEN=los_angeles,TG_ENGINE=torchdnl scripts/sbatch_reduce_validation.sh
#
#SBATCH --job-name=tg_reduceval
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/reduceval_%A_%a.out
#SBATCH --error=logs/reduceval_%A_%a.err

set -euo pipefail

REPO="${TG_REPO:?ABORT: set TG_REPO to the repository checkout}"
VENV="${TG_VENV:?ABORT: set TG_VENV to a Python environment with this package installed}"
: "${TG_SCEN:?ABORT: set TG_SCEN (berlin|los_angeles)}"
case "$TG_SCEN" in
  berlin)      NETWORK=$REPO/tamarl/data/scenarios/berlin/berlin-v6.0-network-with-pt.xml ;;
  los_angeles) NETWORK=$REPO/tamarl/data/scenarios/los_angeles/la-v1.0-network.xml ;;
  *) echo "ABORT: unknown TG_SCEN $TG_SCEN" >&2; exit 1 ;;
esac
# Must match the tree sbatch_validate_berlin.sh wrote, which is per-population so a
# raw-B1 run can never be mistaken for a car-only one.
POP_TAG=${TG_POP:-B1car}
VALROOT=${TG_VALROOT:-$REPO/results/validation_${TG_SCEN}_$POP_TAG}
SEED=${SLURM_ARRAY_TASK_ID}

: "${TG_ENGINE:?ABORT: set TG_ENGINE (matsim|torchdnl)}"

cd "$REPO"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

SEEDDIR=$VALROOT/$TG_ENGINE/seed_$SEED
test -d "$SEEDDIR" || { echo "ABORT: $SEEDDIR does not exist yet" >&2; exit 1; }

# 16 GB for a 544 MB gzipped stream is deliberate headroom, not a requirement:
# the scan itself is O(1) in the event count, and the only large allocation is
# the per-link volume vector (202,939 int64 = 1.6 MB).
# TG_FORCE=1 re-reduces a seed whose reduced.npz already exists. Needed because
# the reducer is idempotent by design, so without it a re-run silently SKIPS and
# still exits 0 -- which would look like a successful re-reduction.
FORCE=()
[ "${TG_FORCE:-0}" = "1" ] && FORCE=(--force)

srun python3 scripts/reduce_validation_events.py \
  --network "$NETWORK" --seed-dir "$SEEDDIR" --engine "$TG_ENGINE" \
  --max-steps "${TG_MAX_STEPS:-129600}" --bucket-sec "${TG_BUCKET_SEC:-900}" \
  --leg-mode "${TG_LEG_MODE:-car}" "${FORCE[@]}"

echo "[ok] reduced $TG_ENGINE seed $SEED"
