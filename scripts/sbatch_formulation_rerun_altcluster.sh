#!/bin/bash
# SLURM array job: the LOS ANGELES half of the MSA / RD od_pair rerun, on a SECOND
# cluster. Companion to scripts/sbatch_formulation_rerun.sh, which runs the Sioux Falls
# and Berlin halves on the primary one.
#
# WHY A SECOND CLUSTER. A long maintenance reservation took every GPU node on the
# primary cluster two days out, so a 20 h job could no longer be placed there
# (`ReqNodeNotAvail, Reserved for maintenance`). The second cluster has no such
# reservation, and it is where the original campaign ran: the 5% population, the
# 9.4 GB top-9 candidate-route cache and the environment are all already in place, so
# nothing has to be staged or re-enumerated.
#
#   sbatch --array=0-9 scripts/sbatch_formulation_rerun_altcluster.sh
#
# WHAT RUNS. Ten cells: {msa, rd} x seeds 1..5, formulation=od_pair, on the
# settings recorded by the original runs in Weights & Biases (June 2026):
# population token 5pct (388,334 agents, 1,699,575 legs), max_steps 129600,
# scale_factor 0.1, 100 episodes, top_k_paths 9, link_tt_interval 300,
# epsilon_ratio 0.10. NOT from the repo's JSON configs, which have drifted:
# they now say max_steps 115200, and train()'s own link_tt_interval default is
# 60 s, which would change MSA's update target and the regret metric.
#
# ENGINE VERSION, DELIBERATE. This checkout is the June engine, i.e. WITHOUT the
# 2026-10-02 single-link-leg fix, exactly as the existing Los Angeles curves
# were produced. The agent -> od_pair contrast is therefore clean within Los
# Angeles, which is the claim being corrected. The cost is that the od_pair arm
# spans two engine versions across scenarios (Sioux Falls and Berlin ran on the
# primary cluster on the fixed engine); that belongs in the paper as one sentence.
# Porting the
# fix here instead would have put a drifted loader in front of a legacy 9.4 GB
# path cache two days before the deadline, for no gain on the question asked.
#
# --time=24:00:00 comes from the measured 19.7-20.3 h of the original runs
# (W&B `_runtime`), not from a bucket. Those ran on a node with 48 cores and
# 540 GB, i.e. an A100 node; an H100 should come in under that.
#
#SBATCH --job-name=tg_form_la
# The Slurm ACCOUNT is not baked in: pass it per submission
# (`sbatch --account=...`, or export SBATCH_ACCOUNT). Allocations differ per site and
# the right one also depends on which of yours has fairshare left.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
# The 2.7 GB population XML is re-parsed by every cell (the original runs did
# not pass --save_pickle either, and the 20 h measurement includes that parse),
# and the legacy top-9 path cache is a 9.4 GB pickle.
#SBATCH --mem=187G
#SBATCH --time=24:00:00
#SBATCH --output=logs/form_%A_%a.out
#SBATCH --error=logs/form_%A_%a.err

set -euo pipefail

REPO="${TG_REPO:?ABORT: set TG_REPO to the repository checkout}"
VENV=${TG_VENV:-$REPO/ENV}
OUTDIR=${TG_OUTDIR:-$REPO/results/formulation_rerun}
EPISODES=${TG_EPISODES:-100}
FORM=${TG_FORM:-od_pair}
SEEDS=(1 2 3 4 5)
METHODS=(msa rd)

SCENDIR=$REPO/tamarl/data/scenarios/los_angeles
POP=5pct
MAX_STEPS=129600
SCALE=0.1

cd "$REPO"
module load python/3.11.5 >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "[gate] node=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || { echo "ABORT: no GPU allocated" >&2; exit 1; }
python3 -c "
import sys, tamarl
if not tamarl.__file__.startswith('$REPO'):
    sys.exit(f'ABORT: imported tamarl from {tamarl.__file__}, expected a path under $REPO')
print('[gate] tamarl from', tamarl.__file__)
"

mkdir -p "$OUTDIR"

IDX="${SLURM_ARRAY_TASK_ID:?ABORT: this is an array job, pass --array=0-9}"
if [ "$IDX" -lt 0 ] || [ "$IDX" -gt 9 ]; then
  echo "ABORT: array index $IDX outside 0-9 (2 methods x 5 seeds)" >&2
  exit 1
fi
METHOD=${METHODS[$((IDX / 5))]}
SEED=${SEEDS[$((IDX % 5))]}
TAG="los_angeles_${METHOD}_${FORM}_s${SEED}"

echo "[cell] idx=$IDX method=$METHOD form=$FORM seed=$SEED tag=$TAG"

srun python3 scripts/run_scale_cell.py \
  --scenario "$SCENDIR" --population "$POP" \
  --method "$METHOD" --seed "$SEED" --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" --scale_factor "$SCALE" \
  --top_k_paths 9 --link_tt_interval 300.0 --epsilon_ratio 0.10 \
  --formulation "$FORM" \
  --arm "formulation_${FORM}" \
  --tag "$TAG" --device cuda \
  --output_dir "$OUTDIR"

echo "[ok] $TAG"
