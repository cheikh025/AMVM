#!/usr/bin/env bash
# Measure two solver builds by alternating them, not by running each in a block.
#
#   experiments/ab_interleaved.sh <baseline-worktree> <repetitions>
#
# Block measurement cannot separate a variant's effect from anything that drifts
# over the run: thermal state, battery charge, a power-source change, background
# load. Alternating cancels any such drift to first order, because both builds
# see the same conditions on average. The power source is recorded per repetition
# so a run that straddles a transition is visible rather than silent.
set -eo pipefail

BASELINE="${1:?usage: ab_interleaved.sh <baseline-worktree> [repetitions]}"
REPS="${2:-4}"
PYTHON="${AMVM_PYTHON:-/tmp/amvm-validation/bin/python}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${HERE}/experiments/results/interleaved"
mkdir -p "${OUT}"

measure() { # tree label instance rows iterations
  ( cd "$1" && "${PYTHON}" experiments/bench_row_solve.py run \
      --instance "${HERE}/experiments/data/$3.pt" --mode speed --rows $4 --seeds 9101 \
      --n 16384 --iterations "$5" --label "$2" --out "${OUT}/$2.json" ) | grep -E "^\[" | sed 's/^/    /'
}

for rep in $(seq 1 "${REPS}"); do
  source="$(pmset -g ps | head -1 | sed -E 's/.*drawing from .([^.]*).*/\1/')"
  charge="$(pmset -g batt | tail -1 | grep -oE '[0-9]+%' | head -1)"
  echo "== repetition ${rep}  power: ${source} ${charge}"
  for instance_rows in ${AMVM_AB_CASES:-"model_decoder_layers_0_self_attn_q_proj:0 1:20" "model_decoder_layers_0_fc2:0 1:20"}; do
    instance="${instance_rows%%:*}"; rest="${instance_rows#*:}"
    rows="${rest%%:*}"; iterations="${rest##*:}"
    echo "  ${instance##*_0_}"
    # Alternate within the repetition, so even a fast drift cannot favour one side.
    measure "${BASELINE}" "baseline-r${rep}-${instance##*_0_}" "${instance}" "${rows}" "${iterations}"
    measure "${HERE}"     "defaults-r${rep}-${instance##*_0_}" "${instance}" "${rows}" "${iterations}"
  done
done
echo INTERLEAVEDDONE
