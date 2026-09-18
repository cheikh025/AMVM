"""Route the batched row solve through the extracted `minviol` package.

`ALNS/batched.py` and `minviol` implement the same search. minviol generalizes
the objective from ``max|Ax - b|`` to the violation of ``lower <= Ax <= upper``,
which a quantization row uses in its equality form (``lower == upper == B_k``).
This adapter exists so that the generalization can be run against the engine it
came from, on real instances and at equal time, instead of being trusted.

``batched.py`` is deliberately kept as the reference implementation rather than
replaced by this shim. ``tests/test_minviol_parity.py`` holds the two engines to
an identical trajectory, and that test is only worth anything while both sides
of it exist.

Two differences are real and are the reason this is a variant rather than a
drop-in:

* minviol also runs a single-variable descent. The engine here has swaps only,
  which preserve the multiset of assigned levels. Set
  ``AMVM_MINVIOL_SINGLE_VARIABLE=0`` to turn it off and reproduce the original
  search exactly.
* minviol stops an instance that reaches a zero objective. For a quantization
  row that means the weights represent the activations exactly, so there is
  nothing left to search for.
"""

import os

import torch

import minviol
from minviol.options import Budget, Options

from ALNS import batched


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


# Whether the minviol path adds the single-variable descent the original lacks.
# On by default because it is the better search; off reproduces the original.
SINGLE_VARIABLE_MOVES = _flag("AMVM_MINVIOL_SINGLE_VARIABLE", True)


def solve(inputs, weights, levels, B_k, seconds, acceptance_policy="linf_l2_nonincrease",
          fixed_mask=None, seed=9101, destroy_rate=0.005, n_filters=100,
          candidate_tile=4096, refresh_every=25):
    """``batched.solve``'s signature, served by minviol. Returns ``(weights, objective)``.

    The bounds are the equality form of the row target, so the violation minimized
    here is ``|A x - B_k|``: exactly the infinity norm the quantization work
    minimizes, with nothing added.
    """
    bounds = B_k.T.contiguous()
    options = Options(
        acceptance=acceptance_policy,
        # The original threshold is absolute, and the quantization instances were
        # tuned around it. A relative one would change every acceptance decision.
        improvement_tol=batched.EPSILON, relative_improvement=False,
        single_variable_moves=SINGLE_VARIABLE_MOVES,
        n_filters=n_filters, destroy_rate=destroy_rate, candidate_tile=candidate_tile,
        refresh_every=refresh_every, seed=seed,
        # Staged pruning is what the original does unconditionally on this path.
        prune_min_constraints=0,
    )
    results = minviol.solve_batch(
        inputs, bounds, bounds, domain=levels, init="given", x0=weights,
        budget=Budget(seconds=seconds), fixed=fixed_mask, options=options,
        n_instances=weights.shape[0])

    solution = torch.stack([result.x_index for result in results])
    objective = torch.tensor([result.max_violation for result in results],
                             device=inputs.device, dtype=inputs.dtype)
    return solution, objective
