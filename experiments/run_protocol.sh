#!/usr/bin/env bash
# Run the standard evaluation protocol for one solver variant.
#
#   experiments/run_protocol.sh <label> [mode ...]
#
# Variant selection comes from the AMVM_* environment flags in ALNS/tuning.py,
# which are read once per process, so every run here is a fresh process:
#
#   AMVM_SKIP_SETTLED_LOCAL_SEARCH=1 experiments/run_protocol.sh skip-settled
#
# Modes default to all three. Results land in experiments/results/<label>/.
# Timing runs must not share the machine with other GPU work.
set -euo pipefail

LABEL="${1:?usage: run_protocol.sh <label> [mode ...]}"
shift || true
MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then MODES=(trajectory speed quality); fi

PYTHON="${AMVM_PYTHON:-/tmp/amvm-validation/bin/python}"
DATA="${AMVM_DATA_DIR:-experiments/data}"
OUT="experiments/results/${LABEL}"
ROWS="${AMVM_BENCH_ROWS:-0 1 2 3}"
N="${AMVM_BENCH_N:-16384}"
ITERATIONS="${AMVM_BENCH_ITERATIONS:-20}"
SECONDS_BUDGET="${AMVM_BENCH_SECONDS:-10}"
QUALITY_SEEDS="${AMVM_BENCH_SEEDS:-9101 9102 9103}"

mkdir -p "${OUT}"

for instance in model_decoder_layers_0_self_attn_q_proj model_decoder_layers_0_fc1 model_decoder_layers_0_fc2; do
  path="${DATA}/${instance}.pt"
  if [ ! -f "${path}" ]; then
    echo "missing ${path}; run experiments/capture_real_instances.py first" >&2
    exit 1
  fi
  for mode in "${MODES[@]}"; do
    case "${mode}" in
      quality) seeds="${QUALITY_SEEDS}" ;;
      *)       seeds="9101" ;;
    esac
    echo "=== ${LABEL} ${instance} ${mode}"
    # shellcheck disable=SC2086
    "${PYTHON}" experiments/bench_row_solve.py run \
      --instance "${path}" --mode "${mode}" --rows ${ROWS} --seeds ${seeds} \
      --n "${N}" --iterations "${ITERATIONS}" --seconds "${SECONDS_BUDGET}" \
      --label "${LABEL}" --out "${OUT}/${instance}_${mode}.json" \
      | grep -E "^\[|^wrote"
  done
done

echo "protocol complete: ${OUT}"
