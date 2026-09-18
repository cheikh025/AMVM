"""Solver options and stopping budget.

Every knob here was an environment variable read once at import time in the
code this was lifted from. That is a benchmark-protocol artifact: it makes a
setting process-global, so two problems in one process cannot be configured
differently, and a test cannot vary one without a subprocess. They are fields on
a frozen dataclass instead, passed per solve.
"""

from dataclasses import dataclass, field

# Number of live intermediates of the exact-evaluation loop (residuals, their
# violations, and their squares), used when sizing a row tile.
LIVE_INTERMEDIATES = 3


@dataclass(frozen=True)
class Budget:
    """When to stop.

    ``feasibility_tol`` is a *stop condition*: a point whose maximum violation is
    at or below it satisfies every constraint as far as the caller is concerned,
    and there is nothing left to find. This is distinct from
    ``Options.improvement_tol``, which decides whether a move counts as an
    improvement. Conflating the two is how a solver ends up either spinning on
    float noise or stopping while still infeasible.
    """

    seconds: float = 10.0
    stop_when_feasible: bool = True
    feasibility_tol: float = 1e-9
    max_iterations: int | None = None


@dataclass(frozen=True)
class Options:
    """Search and kernel-sizing knobs."""

    # Acceptance. "linf" is the only policy defined for general bounds; the two
    # L2 policies are equality-mode features (see violation.py).
    acceptance: str = "linf"

    # A move counts as an improvement when it beats the incumbent by more than
    # this. Relative is right for a library, where the objective's scale is the
    # caller's: the instances this was lifted from range from 0.2 to 9435. The
    # floor matters because a relative threshold vanishes as the violation goes
    # to zero, which would turn the last few moves into float noise.
    improvement_tol: float = 1e-6
    relative_improvement: bool = True
    improvement_floor: float = 1e-12

    # Size of the top-K screening set. In the sparse backend this is not a
    # tuning knob: it is the set the exactness argument quantifies over, and it
    # must exceed the largest column support. Left at None it is chosen for you.
    n_filters: int | None = None

    # Perturbation strength, as a fraction of the variables.
    destroy_rate: float = 0.005

    # Run the single-variable descent alongside swaps. Swaps alone preserve the
    # multiset of assigned levels, so a swap-only search cannot leave a point
    # whose level histogram is wrong -- which is exactly the cold start. Off only
    # to reproduce the swap-only engine this was lifted from.
    single_variable_moves: bool = True

    # Candidate block of the exact stage: constraints x tile floats.
    candidate_tile: int = 4096

    # Incremental residual updates accumulate float error over thousands of
    # accepted moves, so the caches are rebuilt from scratch this often.
    refresh_every: int = 25

    # Bound on the largest intermediate tensor of one pass. The row tile is
    # derived from it, so the same setting adapts to the device and to m.
    memory_budget_mb: int = 256

    # Staged pruning of the exact evaluation. Dense only: splitting a ragged
    # sparse segment costs more than evaluating it.
    prune_first_stage: int = 512
    prune_stage_growth: int = 4

    # Fewest constraints for which staged pruning is worth its overhead. Pruning
    # buys the right to stop reading constraints, so it pays only when there are
    # many left to stop reading; below that, the gather, the per-stage compaction
    # and the extra launches cost more than they save. This is why a library
    # cannot ship one global setting: the instances this engine was lifted from
    # span 192 constraints to 262,144.
    prune_min_constraints: int = 8192

    seed: int = 9101

    def improvement_epsilon(self, objective):
        """Return the improvement threshold for a given incumbent objective.

        Accepts a float or a tensor and returns the same kind, so the caller
        does not have to branch on which it is holding.
        """
        if not self.relative_improvement:
            return self.improvement_tol
        scaled = abs(objective) * self.improvement_tol
        if hasattr(scaled, "clamp_min"):
            return scaled.clamp_min(self.improvement_floor)
        return max(scaled, self.improvement_floor)

    def row_tile(self, n_constraints: int, candidate_chunk: int, element_size: int) -> int:
        """Return the largest row tile whose intermediates fit the memory budget.

        The exact-evaluation loop holds a few ``row_tile x candidate_chunk``
        tensors at once. Sizing from a budget bounds peak memory on any device
        while letting a large-memory GPU use far fewer, far larger launches.
        """
        per_row = max(1, candidate_chunk * element_size * LIVE_INTERMEDIATES)
        affordable = max(1, (self.memory_budget_mb * 1024 * 1024) // per_row)
        return int(max(1, min(n_constraints, affordable)))


def prune_slack(bound):
    """Widen a pruning bound by a hair so ties are never dropped.

    A candidate is dropped when its partial maximum passes the bound, and a
    candidate that exactly ties the incumbent must survive to compete on the
    tie-break. The two sides of that comparison can be computed by different
    kernels, and some backends pick kernels by tensor shape, so a tie can land a
    unit in the last place apart.
    """
    return bound + bound.abs() * 1e-6 + 1e-12
