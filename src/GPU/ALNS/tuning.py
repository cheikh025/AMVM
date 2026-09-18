"""Feature flags and device-aware tile sizes for the ALNS solver.

Defaults are the configuration the measurements in ``docs/gpu-acceleration.md``
recommend: the changes that were exact or that no paired row lost to are on, and
the ones whose margin is inside this device's noise are off. Set a flag to 0 to
get the earlier behaviour back. Benchmarks flip flags through the environment and
run in a separate process, which lets competing implementations coexist on one
commit and keeps every A/B reproducible from a single checkout.

Flags are read once at import time. Set them before the solver is imported.
"""

import os

import torch

TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


# A1: skip a local-search pass whose starting state is already settled, meaning a
# previous pass on that exact state ended without an improving move and nothing
# has changed since. Statistically equivalent, not trajectory identical, because
# the skipped pass would have drawn from the Python RNG.
SKIP_SETTLED_LOCAL_SEARCH = _flag("AMVM_SKIP_SETTLED_LOCAL_SEARCH", True)

# A3/A6: keep the swap pass resident on the device (one synchronization per pass)
# and drop candidates whose running maximum already exceeds the policy bound.
DEVICE_RESIDENT_SWAP = _flag("AMVM_DEVICE_RESIDENT_SWAP", True)

# A6: stop evaluating a candidate once its running maximum has passed the best
# candidate found so far. Selection needs the smallest maximum, so a candidate
# whose partial maximum is already larger can never win and never needs its
# remaining rows. Pruning against the acceptance bound instead was measured and
# discarded: nearly every screening survivor stays under that bound, so it drops
# almost nothing.
PRUNE_TO_INCUMBENT = _flag("AMVM_PRUNE_TO_INCUMBENT", False)

# Stop a local-search descent when the solver's own time budget has expired. The
# descent loops until it finds no improving move, and nothing inside it consults
# the stopping criterion, which is only checked between ALNS iterations. Under a
# tie-accepting policy the descent can keep taking equal-infinity-norm moves for
# far longer than the whole budget: measured on fc1, one 10-second solve was still
# inside its first descent after 180 seconds. Honouring the deadline truncates a
# descent, so it changes the search and must be judged at equal time.
LOCAL_SEARCH_DEADLINE = _flag("AMVM_LOCAL_SEARCH_DEADLINE", True)

# Visit the residual rows in descending order of magnitude inside the pruned
# exact stage, instead of in index order. The objective is a maximum, and the
# audit found it is attained at a single sample with only a handful within 1% of
# it, so the largest residuals decide almost every candidate's score. Visiting
# them first makes each partial maximum nearly its final value after one stage,
# which is what pruning needs to drop candidates early. Selection is unchanged: a
# maximum does not depend on the order it is accumulated in. Only meaningful with
# PRUNE_TO_INCUMBENT, since without pruning every row is visited regardless.
RESIDUAL_ORDERED_ROWS = _flag("AMVM_RESIDUAL_ORDERED_ROWS", False)

# Rows in the first pruning stage. Later stages grow as candidates die, holding
# the intermediate tensor near the memory budget.
PRUNE_FIRST_STAGE = _int("AMVM_PRUNE_FIRST_STAGE", 512)

# How fast a stage may grow. Jumping straight to the memory budget leaves only one
# chance to drop candidates; a bounded growth keeps several cheap chances while
# still reaching large launches for the survivors.
PRUNE_STAGE_GROWTH = _int("AMVM_PRUNE_STAGE_GROWTH", 4)


def prune_slack(bound):
    """Widen a pruning bound by a hair so ties are never dropped.

    A candidate is dropped when its partial maximum passes the bound, and a
    candidate that exactly ties the incumbent must survive to compete on the sum
    of squares. The two sides of that comparison can be computed by different
    kernels, and some backends pick kernels by tensor shape, so a tie can land a
    unit in the last place apart. The slack is far below any difference that
    changes a decision and keeps the tie rule intact.
    """
    return bound + bound.abs() * 1e-6 + 1e-12

# A7, the Gram-matrix L2, has no flag on purpose. It was measured and not
# implemented: the squares accumulation it would replace is about 10% of the
# exact stage. A flag here would advertise a path that does not exist and would
# put a meaningless field in every run manifest. See docs/gpu-acceleration.md.

# B: run rows of a matrix in one batched tensor program instead of one Python
# loop per row.
BATCHED_ROWS = _flag("AMVM_BATCHED_ROWS", False)

# How many rows share one batched program. The residual block is
# rows x samples floats, so this is the main memory knob of the batched path.
BATCH_SIZE = _int("AMVM_BATCH_SIZE", 64)

# Candidate tile of the batched path: the exact-evaluation block is
# samples x tile floats.
BATCH_CANDIDATE_TILE = _int("AMVM_BATCH_CANDIDATE_TILE", 4096)

# A4: derive the row tile from a memory budget instead of using the fixed tile.
# The fixed 256-row tile was chosen for CPU memory; a budget adapts the tile to
# the device and to N, which is what decides how many kernel launches a pass
# costs.
BUDGET_ROW_TILE = _flag("AMVM_BUDGET_ROW_TILE", True)

# Memory budget for the largest intermediate tensor of one swap pass. The row
# tile is derived from it, so the same setting adapts to N and to the device.
SWAP_MEMORY_BUDGET_MB = _int("AMVM_SWAP_MEMORY_BUDGET_MB", 256)

# Baseline tiles, used when a budget-derived tile is not applicable.
VARIABLE_TILE = _int("AMVM_VARIABLE_TILE", 256)
ROW_TILE = _int("AMVM_ROW_TILE", 256)
CANDIDATE_CHUNK = _int("AMVM_CANDIDATE_CHUNK", 512)

# Number of live intermediates of the exact-evaluation loop (residuals, their
# absolute values, and their squares) used when sizing the row tile.
LIVE_INTERMEDIATES = 3


def row_tile_for(n_rows: int, candidate_chunk: int, element_size: int,
                 budget_mb: int = None) -> int:
    """Return the largest row tile whose intermediates fit the memory budget.

    The exact-evaluation loop holds a few ``row_tile x candidate_chunk`` tensors
    at once. Sizing the tile from a budget keeps peak memory bounded on any
    device while letting a large-memory GPU use far fewer, far larger launches
    than the fixed 256-row tile of the baseline.
    """
    budget = SWAP_MEMORY_BUDGET_MB if budget_mb is None else budget_mb
    per_row = max(1, candidate_chunk * element_size * LIVE_INTERMEDIATES)
    affordable = max(1, (budget * 1024 * 1024) // per_row)
    return int(max(1, min(n_rows, affordable)))


def resolve_row_tile(n_rows: int, candidate_chunk: int, element_size: int) -> int:
    """Return the row tile the active configuration asks for."""
    if BUDGET_ROW_TILE:
        return row_tile_for(n_rows, candidate_chunk, element_size)
    return min(ROW_TILE, max(1, n_rows))


def describe(device: torch.device = None) -> dict:
    """Return the active configuration, for run manifests and ledger entries."""
    return {
        "skip_settled_local_search": SKIP_SETTLED_LOCAL_SEARCH,
        "device_resident_swap": DEVICE_RESIDENT_SWAP,
        "prune_to_incumbent": PRUNE_TO_INCUMBENT,
        "prune_first_stage": PRUNE_FIRST_STAGE,
        "prune_stage_growth": PRUNE_STAGE_GROWTH,
        "residual_ordered_rows": RESIDUAL_ORDERED_ROWS,
        "local_search_deadline": LOCAL_SEARCH_DEADLINE,
        "batched_rows": BATCHED_ROWS,
        "batch_size": BATCH_SIZE,
        "batch_candidate_tile": BATCH_CANDIDATE_TILE,
        "budget_row_tile": BUDGET_ROW_TILE,
        "swap_memory_budget_mb": SWAP_MEMORY_BUDGET_MB,
        "variable_tile": VARIABLE_TILE,
        "row_tile": ROW_TILE,
        "candidate_chunk": CANDIDATE_CHUNK,
        "device": None if device is None else str(device),
    }
