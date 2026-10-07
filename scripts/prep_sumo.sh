#!/bin/bash
# Build the SUMO side of the scaling benchmark for one scenario.
#
# Two stages, because they have very different costs:
#   TG_STAGE=net     netconvert the network once      (minutes)
#   TG_STAGE=routes  convert one loading's population (up to hours at the top)
#
# Submit the routes stage as an array, one task per loading, after the net stage:
#   NET=$(sbatch --parsable --export=ALL,TG_SCEN=berlin,TG_STAGE=net scripts/prep_sumo.sh)
#   sbatch --dependency=afterok:$NET --array=0-7 --export=ALL,TG_SCEN=berlin,TG_STAGE=routes scripts/prep_sumo.sh
#
#SBATCH --job-name=tg_prepsumo
# The Slurm ACCOUNT is not baked in: pass it per submission
# (`sbatch --account=...`, or export SBATCH_ACCOUNT). Allocations differ per site and
# the right one also depends on which of yours has fairshare left.
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/prepsumo_%A_%a.out
#SBATCH --error=logs/prepsumo_%A_%a.err

set -euo pipefail

REPO="${TG_REPO:?ABORT: set TG_REPO to the repository checkout}"
PRE="${TG_SUMO_PRELOAD:?ABORT: set TG_SUMO_PRELOAD to libatomic.so.1 on this site}"
NETCONVERT="${TG_NETCONVERT:?ABORT: set TG_NETCONVERT to the netconvert binary}"

: "${TG_SCEN:?ABORT: set TG_SCEN}"
: "${TG_STAGE:?ABORT: set TG_STAGE (net|routes)}"

case "$TG_SCEN" in
  berlin)      SCENDIR=$REPO/tamarl/data/scenarios/berlin;      NET=$SCENDIR/berlin-v6.0-network-with-pt.xml ;;
  los_angeles) SCENDIR=$REPO/tamarl/data/scenarios/los_angeles; NET=$SCENDIR/la-v1.0-network.xml ;;
  *) echo "ABORT: unknown scenario $TG_SCEN" >&2; exit 1 ;;
esac

VENV="${TG_VENV:?ABORT: set TG_VENV to a Python environment with this package installed}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
cd "$REPO"
mkdir -p "$SCENDIR/sumo"

if [ "$TG_STAGE" = "net" ]; then
  CARNET=$SCENDIR/sumo/network-car-modes.xml
  echo "=== normalise les modes avant conversion ==="
  # netconvert's MATSim import turns modes="ride,car,freight" into
  # disallow="all" -- `ride` and `freight` are not SUMO vClasses and it gives up
  # on the whole list rather than keeping `car`. On Berlin that silently
  # forbade 188,836 of 329,361 lanes, i.e. the entire road network, and SUMO
  # then refused every vehicle's first edge. Declaring plain modes="car" makes
  # netconvert emit allow="passenger", and it is also exactly the network the
  # other two engines see: TorchDNL reads car links only and MATSim runs with
  # transit off.
  python3 - "$NET" "$CARNET" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
pat = re.compile(r'modes="([^"]*)"')
n_car = n_other = 0
with open(src, errors="replace") as fi, open(dst, "w") as fo:
    for line in fi:
        m = pat.search(line)
        if m:
            modes = [x.strip() for x in m.group(1).split(",")]
            if "car" in modes:
                line = pat.sub('modes="car"', line, count=1)
                n_car += 1
            else:
                n_other += 1
        fo.write(line)
print(f"links carrying car: {n_car} (rewritten to modes=car); other links: {n_other}")
PY
  NET_OUT=$SCENDIR/sumo/network.net.xml
  CONNS=$SCENDIR/sumo/restored-connections.xml

  # Which route files carry the full set of distinct movements. Berlin's ladder is
  # built by cloning ONE base population, so B1 contains every route the other
  # loadings use; LA ships five independently routed samples, so all five count.
  case "$TG_SCEN" in
    berlin)      ROUTE_ARGS=(--routes "$SCENDIR/sumo/B1${TG_POP_SUFFIX-car}.rou.xml") ;;
    los_angeles) ROUTE_ARGS=(--routes "$SCENDIR/sumo/L0p001${TG_POP_SUFFIX-car}.rou.xml"
                             --routes "$SCENDIR/sumo/L0p01${TG_POP_SUFFIX-car}.rou.xml"
                             --routes "$SCENDIR/sumo/L0p05${TG_POP_SUFFIX-car}.rou.xml"
                             --routes "$SCENDIR/sumo/L0p1${TG_POP_SUFFIX-car}.rou.xml"
                             --routes "$SCENDIR/sumo/L0p25${TG_POP_SUFFIX-car}.rou.xml") ;;
  esac
  for f in "${ROUTE_ARGS[@]}"; do
    [ "$f" = "--routes" ] && continue
    test -f "$f" || { echo "ABORT: missing route file $f; run TG_STAGE=routes first" >&2; exit 1; }
  done

  echo "=== netconvert (passe 1) ==="
  # --no-turnarounds.geometry defaults to TRUE, which suppresses U-turns at
  # two-legged junctions. MATSim's queue model has no turn restrictions at all, so
  # 91.5% of the movements SUMO rejected on Berlin were exactly those U-turns.
  LD_PRELOAD=$PRE "$NETCONVERT" --matsim-files "$CARNET" \
    -o "$NET_OUT" --no-warnings --no-turnarounds.geometry false
  test -f "$NET_OUT" || { echo "ABORT: netconvert produced nothing" >&2; exit 1; }

  echo "=== les routes MATSim doivent etre franchissables dans le reseau SUMO ==="
  if ! python3 scripts/check_sumo_route_connectivity.py --net "$NET_OUT" "${ROUTE_ARGS[@]}"; then
    echo "=== passe 2: on restaure les mouvements elagues ==="
    # netconvert's remaining pruning is lane- and angle-heuristic; the movements it
    # drops are legal in both other engines. Restore exactly the ones the demand
    # uses, so the three engines simulate identical routes.
    python3 scripts/check_sumo_route_connectivity.py --net "$NET_OUT" "${ROUTE_ARGS[@]}" \
      --emit-connections "$CONNS"
    LD_PRELOAD=$PRE "$NETCONVERT" --matsim-files "$CARNET" \
      -o "$NET_OUT" --no-warnings --no-turnarounds.geometry false \
      --connection-files "$CONNS"
    # A residual handful survives: netconvert drops some U-turns even when the
    # connection file declares them explicitly, and no --no-turnarounds.* setting
    # recovers them. On Berlin that is 32 movements out of 5,712 originally missing.
    # Rather than regenerate every engine's population to excise those trips, we
    # measure how much demand SUMO loses to them and let it skip exactly those
    # vehicles, with a hard tolerance so the number can never grow unnoticed.
    python3 scripts/check_sumo_route_connectivity.py --net "$NET_OUT" "${ROUTE_ARGS[@]}" \
      --max-affected-frac "${TG_SUMO_TOLERANCE:-0.001}" \
      | tee "$SCENDIR/sumo/route-connectivity.txt"
    test "${PIPESTATUS[0]}" -eq 0 \
      || { echo "ABORT: trop de demande perdue cote SUMO" >&2; exit 1; }
  fi

  echo "=== gate: le reseau routier doit etre franchissable ==="
  DIS=$(grep -c 'disallow="all"' "$NET_OUT" || true)
  ALLOW=$(grep -c 'allow="passenger' "$NET_OUT" || true)
  EDGES=$(grep -c '<edge id=' "$NET_OUT")
  echo "  aretes=$EDGES  voies allow=passenger=$ALLOW  voies disallow=all=$DIS"
  [ "$ALLOW" -gt 0 ] || { echo "ABORT: aucune voie ouverte aux voitures" >&2; exit 1; }
  ls -lh "$NET_OUT" | awk '{print "  taille:", $5}'
  echo DONE
  exit 0
fi

# --- routes stage -----------------------------------------------------------
if ! ASSIGN="$(python3 scripts/bench_grid.py --scenario "$TG_SCEN" --engine sumo \
               --index $(( SLURM_ARRAY_TASK_ID * 3 )))"; then
  echo "ABORT: grid mapper refused index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
eval "$ASSIGN"
echo "[grid] loading=$CELL_LOADING capacity=$CELL_CAPACITY_FACTOR pop=$CELL_POP_TAG"

POP_SUFFIX=${TG_POP_SUFFIX-car}
OUT=$SCENDIR/sumo/${CELL_POP_TAG}${POP_SUFFIX}.rou.xml
if [ -f "$OUT" ]; then echo "[skip] $OUT already built"; exit 0; fi

# Routes are built from the ORIGINAL MATSim network so link ids match the ones
# netconvert preserved verbatim as SUMO edge ids.
python3 scripts/convert_fixed_routes_to_sumo.py \
  --network "$NET" \
  --population "$SCENDIR/${CELL_POP_TAG}${POP_SUFFIX}_population.xml" \
  --capacity-factor "$CELL_CAPACITY_FACTOR" \
  --output "$OUT"
N=$(grep -c '<vehicle ' "$OUT" || true)
echo "  vehicules=$N  taille=$(ls -lh "$OUT" | awk '{print $5}')"
[ "$N" -gt 0 ] || { echo "ABORT: aucun vehicule ecrit" >&2; exit 1; }
echo DONE
