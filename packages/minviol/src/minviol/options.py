"""Solver options and stopping budget.

Every knob here was an environment variable read once at import time in the
code this was lifted from. That is a benchmark-protocol artifact: it makes a
setting process-global, so two problems in one process cannot be configured
differently, and a test cannot vary one without a subprocess. They are fields on
a frozen dataclass instead, passed per solve.
"""

from dataclasses import dataclass, field

import torch

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
    feasibility_tol: float | None = None
    max_iterations: int | None = None

    def resolve_feasibility_tol(self, dtype, scale, n_constraints) -> float:
        """The violation below which a point counts as satisfying the constraints.

        Left unset it is derived, because a fixed small number is the wrong answer
        in float32. Computing ``A x`` over m terms accumulates roughly
        ``sqrt(m) * eps`` of relative error, so on 100,000 float32 constraints
        with bounds of order 100 the arithmetic simply cannot deliver a residual
        below about 4e-3 -- and a solver that never reports success because its
        own dtype cannot reach its own threshold is worse than useless. The same
        formula on float64 gives about 7e-12, so precision is not thrown away
        where it exists.

        Pass a number to say exactly what your problem counts as satisfied.
        """
        if self.feasibility_tol is not None:
            return float(self.feasibility_tol)
        eps = torch.finfo(dtype).eps
        return float(max(1.0, float(scale)) * eps * max(1.0, n_constraints ** 0.5))


@dataclass(frozen=True)
class Options:
    """Search and kernel-sizing knobs.

    The defaults are the configuration measured best for general constraint
    systems in `experiments/algo_memory.md`: single-variable moves only, a kick
    aimed at the worst constraint that widens while an instance stalls, and an
    acceptance rule that lets a move which leaves the maximum alone win on the
    tie-break. Against the original swap-based search that is 30 of 30 paired
    comparisons better, with the feasibility count going from 0 of 30 to 17 of 30.

    A quantization-shaped problem wants the opposite of most of this -- it starts
    near round-to-nearest with the level histogram already right, which is where a
    swap is a strong move. Set `swap_moves=True` and `perturbation="random"` for
    that; `src/GPU/ALNS/minviol_engine.py` in the AMVM repository does exactly so.
    """

    # Acceptance. "linf" is the only policy defined for general bounds; the two
    # L2 policies are equality-mode features (see violation.py).
    acceptance: str = "linf_l2_tiebreak"

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

    # How the search kicks itself out of a local optimum.
    #   "random" -- move a random subset of variables by one level, in a random
    #               direction. The original engine's perturbation.
    #   "active" -- move a random subset of the variables appearing in the
    #               currently worst constraint, each in the direction that
    #               relieves that constraint.
    # The objective is a maximum, so it is attained at one constraint at a time;
    # a kick that misses that constraint cannot change the objective, and with n
    # variables a blind kick misses it with probability 1 - |support|/n.
    perturbation: str = "active"

    # Widen the kick while an instance keeps failing to beat its incumbent, and
    # snap back as soon as it succeeds. A fixed one-variable kick is undone by
    # the descent every time on some instances -- measured at 0 accepted out of
    # 40 -- and no choice of which variable fixes that; the basin is simply wider
    # than the kick. The schedule doubles the kick every `kick_patience`
    # consecutive failures, up to `kick_max_rate` of the variables.
    kick_escalation: bool = True
    kick_patience: int = 5
    kick_max_rate: float = 0.25

    # Run the single-variable descent alongside swaps. Swaps alone preserve the
    # multiset of assigned levels, so a swap-only search cannot leave a point
    # whose level histogram is wrong -- which is exactly the cold start. Off only
    # to reproduce the swap-only engine this was lifted from.
    single_variable_moves: bool = True

    # Run the swap descent, which exchanges the levels of two variables. It is a
    # strong move where the level histogram is roughly right to begin with, which
    # is the quantization case. It is also the expensive move for a sparse matrix,
    # since each candidate touches two columns and the pass is bound by kernel
    # launches rather than by arithmetic -- see docs/sparse-design.md.
    swap_moves: bool = False

    # Move two variables at once, with independent steps, drawn from the single
    # moves that lose the least. Runs only for instances no other pass could move,
    # because it costs a full single-candidate scoring before it starts. This is
    # the move a swap only approximates: a swap fixes the two steps to be equal
    # and opposite, which preserves the level histogram and has nothing to do with
    # which constraint is binding.
    compound_moves: bool = False

    # How many of the least damaging single candidates the pairs are drawn from.
    # All pairs would be quadratic in the variable count, and a pair built from
    # two badly damaging moves is not going to win.
    compound_width: int = 32

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
