#!/bin/bash
# SLURM array job: re-run the two POPULATION baselines (MSA, Replicator
# Dynamics) of the paper's routing-performance campaign under
# formulation="od_pair".
#
# WHY. Every paper config sets formulation="agent", i.e. one parameter block per
# trip leg, and both of these methods are population methods that degenerate on
# a one-vehicle block:
#   * RD's update becomes the identity. The chosen path's utility is
#     exp(beta*0)=1 after per-block max-normalisation, the unchosen ones are
#     imputed at u_bar_observed=1, so new_probs = p*1/1 = p. Measured:
#     max|p_50 - p_0| = 0.0 over 50 episodes at agent level, 0.111 at OD level.
#   * MSA loses its flow averaging. Its target is the share of the block's
#     vehicles whose best path is k, which on a one-vehicle block is 0 or 1, so
#     the update is p <- (1-a)p + a*onehot(best) per leg and no flow pattern is
#     ever averaged.
# Only these two methods and only these three scenarios are affected; the other
# seven policies are per-vehicle learners and their existing curves stand.
#
# EVERY OTHER KNOB MATCHES THE ORIGINAL CAMPAIGN, which ran on another cluster in
# June 2026. Taken from those runs' recorded Weights & Biases configs, not
# from the JSON files in the repo, which have drifted:
#   sioux_falls      no population filter (the routed base: 65,758 agents),
#                    max_steps 86400,  scale_factor 1.0
#   berlin-100pct    population token 100pct (957,040 agents),
#                    max_steps 129600, scale_factor 1.0
#   los_angeles-5pct population token 5pct (388,334 agents),
#                    max_steps 129600, scale_factor 0.1
#   all three: 100 episodes, top_k_paths 9, penalty enumeration f=1.5,
#              link_tt_interval 300 s, epsilon_ratio 0.10, seeds 1..5.
# NOTE the two that are easy to get wrong: max_steps is 129600 (36 h) on the
# metropolitan scenarios -- the committed configs say 115200 -- and
# link_tt_interval is 300 s, against train()'s own default of 60 s. MSA reads
# its update target off that binning, so 60 s would not be the same method.
#
# Submit ONE ARRAY PER SCENARIO: the array shares a single --time, and an
# episode on Berlin or Los Angeles costs two orders of magnitude more than one
# on Sioux Falls.
#
#   # caches first: parse the population, enumerate the top-9 candidate routes
#   sbatch --time=3:00:00 --export=ALL,TG_PREP=1,TG_SCEN=sioux_falls  scripts/sbatch_formulation_rerun.sh
#   sbatch --time=12:00:00 --export=ALL,TG_PREP=1,TG_SCEN=berlin      scripts/sbatch_formulation_rerun.sh
#   sbatch --time=12:00:00 --export=ALL,TG_PREP=1,TG_SCEN=los_angeles scripts/sbatch_formulation_rerun.sh
#
#   # then the cells, dependent on their scenario's prep job
#   sbatch --time=12:00:00 --array=0-9 --dependency=afterok:<prep_sf> \
#          --export=ALL,TG_SCEN=sioux_falls  scripts/sbatch_formulation_rerun.sh
#   sbatch --time=3-00:00  --array=0-9 --dependency=afterok:<prep_berlin> \
#          --export=ALL,TG_SCEN=berlin       scripts/sbatch_formulation_rerun.sh
#   sbatch --time=3-00:00  --array=0-9 --dependency=afterok:<prep_la> \
#          --export=ALL,TG_SCEN=los_angeles  scripts/sbatch_formulation_rerun.sh
#
# TG_FORM defaults to od_pair. Set TG_FORM=agent to submit the control arm on
# the same engine: the existing agent-level curves were produced before the
# single-link-leg fix, so without that arm a change in the MSA/RD curves cannot
# be attributed to the formulation alone.
#
#SBATCH --job-name=tg_form
# The Slurm ACCOUNT is not baked in: pass it per submission
# (`sbatch --account=...`, or export SBATCH_ACCOUNT). Allocations differ per site and
# the right one also depends on which of yours has fairshare left.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
# Berlin's 100% plans are a 3.3 GB XML and Los Angeles's 5% sample a 2.7 GB one;
# the parse, not the simulation, is what needs the memory. The bench campaign
# got through the same files on 64G, so this is that with headroom.
#SBATCH --mem=96G
#SBATCH --output=logs/form_%A_%a.out
#SBATCH --error=logs/form_%A_%a.err

set -euo pipefail

REPO="${TG_REPO:?ABORT: set TG_REPO to the repository checkout}"
VENV="${TG_VENV:?ABORT: set TG_VENV to a Python environment with this package installed}"
OUTDIR=${TG_OUTDIR:-$REPO/results/formulation_rerun}
EPISODES=${TG_EPISODES:-100}
FORM=${TG_FORM:-od_pair}
SEEDS=(1 2 3 4 5)
METHODS=(msa rd)

: "${TG_SCEN:?ABORT: set TG_SCEN to sioux_falls, berlin or los_angeles}"

case "$TG_SCEN" in
  sioux_falls)
    SCENDIR=$REPO/tamarl/data/scenarios/sioux_falls
    POP=Siouxfalls          # the routed base file, as the original runs used
    MAX_STEPS=86400
    SCALE=1.0
    ;;
  berlin)
    SCENDIR=$REPO/tamarl/data/scenarios/berlin
    POP=100pct
    MAX_STEPS=129600
    SCALE=1.0
    ;;
  los_angeles)
    SCENDIR=$REPO/tamarl/data/scenarios/los_angeles
    POP=L0p05               # the 5% sample; the original campaign called it "5pct"
    MAX_STEPS=129600
    SCALE=0.1               # capacities rescaled to the sample, as in the paper
    ;;
  *) echo "ABORT: unknown scenario $TG_SCEN" >&2; exit 1 ;;
esac

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

# --- Prep mode: one cheap episode, purely to write the caches ----------------
# The parsed ScenarioData pickle and the top-9 candidate-route cache are both
# written next to the scenario and are shared by all ten cells. Building them
# once here keeps nine jobs from each spending the same tens of minutes on the
# same parse, and keeps ten jobs from writing the same cache file at once.
if [ -n "${TG_PREP:-}" ]; then
  srun python3 scripts/run_scale_cell.py \
    --scenario "$SCENDIR" --population "$POP" \
    --method random --seed 0 --episodes 1 \
    --max_steps "$MAX_STEPS" --scale_factor "$SCALE" \
    --top_k_paths 9 --link_tt_interval 300.0 --epsilon_ratio 0.10 \
    --formulation "$FORM" --save_pickle \
    --tag "prep_${TG_SCEN}" --device cuda \
    --output_dir "$OUTDIR/prep"
  echo "[ok] prep ${TG_SCEN}"
  exit 0
fi

# --- One cell: index -> (method, seed) ---------------------------------------
IDX="${SLURM_ARRAY_TASK_ID:?ABORT: this is an array job, pass --array=0-9}"
if [ "$IDX" -lt 0 ] || [ "$IDX" -gt 9 ]; then
  echo "ABORT: array index $IDX outside 0-9 (2 methods x 5 seeds)" >&2
  exit 1
fi
METHOD=${METHODS[$((IDX / 5))]}
SEED=${SEEDS[$((IDX % 5))]}
TAG="${TG_SCEN}_${METHOD}_${FORM}_s${SEED}"

echo "[cell] idx=$IDX scen=$TG_SCEN method=$METHOD form=$FORM seed=$SEED tag=$TAG"

srun python3 scripts/run_scale_cell.py \
  --scenario "$SCENDIR" --population "$POP" \
  --method "$METHOD" --seed "$SEED" --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" --scale_factor "$SCALE" \
  --top_k_paths 9 --link_tt_interval 300.0 --epsilon_ratio 0.10 \
  --formulation "$FORM" --save_pickle \
  --arm "formulation_${FORM}" \
  --tag "$TAG" --device cuda \
  --output_dir "$OUTDIR"

echo "[ok] $TAG"
