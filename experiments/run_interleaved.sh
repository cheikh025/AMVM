#!/usr/bin/env bash
# Run the standard protocol for several variants at once, interleaved.
#
#   experiments/run_interleaved.sh "defaults" "all-off SKIP_SETTLED_LOCAL_SEARCH=0 DEVICE_RESIDENT_SWAP=0"
#
# Each argument is one variant: a label, then tuning attributes to set for it.
# Every other flag is restored to the file default before each solve, so a
# variant is exactly its named settings.
#
# This costs no extra solves. It runs the same ones as calling run_protocol.sh
# once per variant, reordered so that every (row, seed) point is measured for all
# variants back to back. Drift over the run then hits every variant equally
# instead of whichever one happened to be running, which is what made the first
# round of Tier A timings 6% to 8% too generous. It also launches one process per
# instance and mode for all variants together, rather than one per variant, so it
# imports torch and loads each instance fewer times than the old protocol.
#
# Use experiments/ab_interleaved.sh instead when the two sides are different
# commits, which flags cannot express; that one needs a worktree and does pay for
# a second set of processes.
#
# IMPORTANT: this is only safe for variants that run the SAME kernels. Sharing a
# process also shares compiled-kernel caches, so a variant whose tensor shapes
# differ from its neighbour's can be flattered by whatever ran before it. The
# pruned swap stage compacts its candidate set every stage and so compiles a
# kernel per shape: measured this way it looked 1.25x faster than the unpruned
# path, against 1.01x with one variant per process. When variants differ in shape
# behaviour, run one per process and alternate the processes instead.
#
# Results land in experiments/results/<label>/, the same layout as
# run_protocol.sh, so summarize_results.py and `bench_row_solve.py compare` read
# them unchanged.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: run_interleaved.sh \"<label> [ATTR=VALUE ...]\" ..." >&2
  exit 1
fi

PYTHON="${AMVM_PYTHON:-/tmp/amvm-validation/bin/python}"
DATA="${AMVM_DATA_DIR:-experiments/data}"
RESULTS="${AMVM_RESULTS_DIR:-experiments/results}"
ROWS="${AMVM_BENCH_ROWS:-0 1 2 3}"
N="${AMVM_BENCH_N:-16384}"
ITERATIONS="${AMVM_BENCH_ITERATIONS:-20}"
SECONDS_BUDGET="${AMVM_BENCH_SECONDS:-10}"
QUALITY_SEEDS="${AMVM_BENCH_SEEDS:-9101 9102 9103}"
INSTANCES="${AMVM_BENCH_INSTANCES:-model_decoder_layers_0_self_attn_q_proj model_decoder_layers_0_fc1 model_decoder_layers_0_fc2}"
MODES="${AMVM_BENCH_MODES:-trajectory speed quality}"

VARIANTS=()
for spec in "$@"; do
  # Deliberately unquoted: one spec expands to "--variant LABEL ATTR=VALUE ...".
  # shellcheck disable=SC2206
  VARIANTS+=(--variant ${spec})
done

# A timing run on battery is not comparable with one on wall power: this machine
# uses a different performance mode for each. Interleaving cancels drift between
# variants, but absolute numbers still deserve the warning. No pmset means a
# non-Apple host, where this does not apply.
if command -v pmset >/dev/null 2>&1; then
  if ! pmset -g ps | head -1 | grep -q "AC Power"; then
    echo "warning: running on battery; absolute timings are not comparable with wall-power runs" >&2
  fi
fi

for instance in ${INSTANCES}; do
  path="${DATA}/${instance}.pt"
  if [ ! -f "${path}" ]; then
    echo "missing ${path}; run experiments/capture_real_instances.py first" >&2
    exit 1
  fi
  for mode in ${MODES}; do
    case "${mode}" in
      quality) seeds="${QUALITY_SEEDS}" ;;
      *)       seeds="9101" ;;
    esac
    echo "=== ${instance} ${mode} (${#VARIANTS[@]} variant arguments, interleaved)"
    # shellcheck disable=SC2086
    "${PYTHON}" experiments/bench_row_solve.py run \
      --instance "${path}" --mode "${mode}" --rows ${ROWS} --seeds ${seeds} \
      --n "${N}" --iterations "${ITERATIONS}" --seconds "${SECONDS_BUDGET}" \
      --results-dir "${RESULTS}" "${VARIANTS[@]}" \
      | grep -E "^\[|^wrote"
  done
done

echo "interleaved protocol complete: ${RESULTS}"
