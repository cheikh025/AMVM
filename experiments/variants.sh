#!/usr/bin/env bash
# The variant stack under evaluation, defined once so every run is reproducible.
#
# Each entry is cumulative: it adds one change to the entry above it, which is
# what makes the difference between two rows attributable to that one change.
# "baseline" is not here: it is measured from a pristine worktree at the commit
# the branch started from, because the first change (host-side index gathering)
# has no flag and is always on in this tree.
#
#   source experiments/variants.sh
#   for name in ${VARIANT_ORDER}; do env $(variant_flags "${name}") ... ; done
#
# Written for bash 3.2, the version macOS ships, so no associative arrays.

VARIANT_ORDER="a2-host-indexing a1-skip-settled a3-device-resident a4-budget-tile a6-prune-incumbent"

variant_flags() {
  case "$1" in
    # Every entry names every flag, so the ladder keeps reproducing the same
    # configurations now that the recommended ones are the defaults.
    #
    # A2: gather removed indices and levels in one transfer instead of per element.
    a2-host-indexing)
      echo "AMVM_SKIP_SETTLED_LOCAL_SEARCH=0 AMVM_DEVICE_RESIDENT_SWAP=0 AMVM_BUDGET_ROW_TILE=0" ;;
    # A1: skip a local-search pass whose starting point is already settled.
    a1-skip-settled)
      echo "AMVM_SKIP_SETTLED_LOCAL_SEARCH=1 AMVM_DEVICE_RESIDENT_SWAP=0 AMVM_BUDGET_ROW_TILE=0" ;;
    # A3: keep the swap pass on the device, one synchronization per pass.
    a3-device-resident)
      echo "AMVM_SKIP_SETTLED_LOCAL_SEARCH=1 AMVM_DEVICE_RESIDENT_SWAP=1 AMVM_BUDGET_ROW_TILE=0" ;;
    # A4: size the row tile from a memory budget instead of the fixed 256 rows.
    # This is the default configuration.
    a4-budget-tile)
      echo "AMVM_SKIP_SETTLED_LOCAL_SEARCH=1 AMVM_DEVICE_RESIDENT_SWAP=1 AMVM_BUDGET_ROW_TILE=1" ;;
    # A6: drop candidates that can no longer beat the best one found so far.
    # The candidate chunk stays at its default: raising it was measured and is a
    # regression on both layers, with or without pruning.
    a6-prune-incumbent)
      echo "AMVM_SKIP_SETTLED_LOCAL_SEARCH=1 AMVM_DEVICE_RESIDENT_SWAP=1 AMVM_BUDGET_ROW_TILE=1 AMVM_PRUNE_TO_INCUMBENT=1" ;;
    *)
      echo "unknown variant: $1" >&2; return 1 ;;
  esac
}
