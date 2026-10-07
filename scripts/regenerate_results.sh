#!/usr/bin/env bash
# Rebuild RESULTS.md from the CSVs committed under results/.
#
# This is the exact invocation that produced the committed RESULTS.md. It reads only
# files that are in the repository, so it runs anywhere with no cluster access and no
# scenario data — useful as a check that every number in RESULTS.md traces to a CSV.
#
#     ./scripts/regenerate_results.sh
#     git diff --exit-code RESULTS.md     # should be empty
#
# The CSVs themselves come off the cluster via scripts/anonymize_results.py.
#
# RESULTS.md is generated tables only: every line of it is derived from a committed CSV
# or validation summary. There is no hand-written prose to preserve, so the file can be
# regenerated from scratch at any time.

set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/make_results_archive.py \
  --bench "Berlin — 159,039 car links (202,939 total)=results/benchmarks/berlin" \
  --bench "Los Angeles — 1,154,727 car links (1,175,146 total)=results/benchmarks/los_angeles" \
  --bench "Sioux Falls — 76 links=results/benchmarks/sioux_falls" \
  --bench "Berlin, superseded: raw population ladder, pre-fix engine=results/benchmarks/superseded/berlin_raw_prefix" \
  --bench "Los Angeles, superseded: raw population ladder, pre-fix engine=results/benchmarks/superseded/los_angeles_raw_prefix" \
  --fidelity "Berlin, 1% car-only sample, 20 seeds=results/validation/berlin_1pct_car/summary.md" \
  --fidelity "Los Angeles, 1% car-only sample, 20 seeds=results/validation/los_angeles_1pct_car/summary.md" \
  --fidelity "Berlin, same data through the pre-fix engine (superseded)=results/validation/berlin_1pct_car_prefix_engine/summary.md" \
  --output RESULTS.md
