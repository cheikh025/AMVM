"""What the tie-break power is allowed to change, and what it must not.

The tie-break settles candidates that leave the maximum where it is. Summing the
squared violation over every constraint makes it a potential function, which is
what makes a plateau walk terminate; summing over the worst constraints instead
does not, and cycles (experiment 7). Raising the power is the way to weight the
worst constraints without moving the set the sum runs over, so these tests pin
the two properties that argument rests on: the sum stays over a fixed set, and it
stays additive over constraints so the sparse backend may carry it incrementally.
"""
import pytest
import torch

import minviol
from minviol import Budget, Options
from minviol.backends import DenseMatrix, SparseMatrix
from minviol.counters import Counters
from minviol import engine
# By name: `minviol.violation` is the re-exported function, which shadows the module.
from minviol.violation import (LINF_L2_NONINCREASE, LINF_L2_TIEBREAK,
                               tiebreak_terms)
from minviol.problem import Batch


def system(n_constraints=120, n_variables=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_constraints, n_variables, generator=g)
    domain = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    x_star = domain[torch.randint(len(domain), (n_variables,), generator=g)]
    return A, domain, A @ x_star


def test_power_two_is_the_sum_of_squares_bit_for_bit():
    """The control has to be the incumbent, or the comparison measures the rewrite."""
    v = torch.rand(500, 7) * 40.0
    assert torch.equal(tiebreak_terms(v, 2.0), v.square())


@pytest.mark.parametrize("power", [2.0, 4.0, 8.0])
def test_tiebreak_is_additive_over_constraints(power):
    """The sparse backend corrects only the touched rows; that needs additivity."""
    v = torch.rand(64, 3) * 10.0
    scale = torch.full((3,), 7.0)
    whole = tiebreak_terms(v, power, scale).sum(dim=0)
    parts = (tiebreak_terms(v[:20], power, scale).sum(dim=0)
             + tiebreak_terms(v[20:], power, scale).sum(dim=0))
    assert torch.allclose(whole, parts, rtol=1e-5)


@pytest.mark.parametrize("power", [4.0, 8.0])
def test_scale_is_fixed_for_the_solve(power):
    """A scale that tracked the point would be a new potential function each step."""
    A, domain, target = system()
    x0 = torch.zeros(1, A.shape[1], dtype=torch.long)
    bounds = target[:, None]
    batch = Batch(DenseMatrix(A), x0, domain[None, :], bounds, bounds,
                  acceptance=LINF_L2_TIEBREAK, tiebreak_power=power)
    captured = batch.tiebreak_scale.clone()
    assert captured is not None

    batch.x_idx[0, 0] = 2
    batch.refresh()
    assert torch.equal(batch.tiebreak_scale, captured)


@pytest.mark.parametrize("power", [2.0, 4.0, 8.0])
def test_dense_and_sparse_score_candidates_alike(power):
    """Hard constraint 4: the backends must agree, at every power."""
    A, domain, target = system(n_constraints=90, n_variables=20, seed=3)
    x0 = torch.zeros(1, A.shape[1], dtype=torch.long)
    bounds = target[:, None]
    options = Options(prune_min_constraints=0, tiebreak_power=power,
                      acceptance=LINF_L2_TIEBREAK)

    scores = []
    for matrix in (DenseMatrix(A), SparseMatrix.from_dense(A)):
        batch = Batch(matrix, x0, domain[None, :], bounds, bounds,
                      acceptance=LINF_L2_TIEBREAK, tiebreak_power=power)
        rows = engine.screening_rows(batch, 64)
        tag, left, q_left = engine.candidate_moves(
            batch, torch.ones(1, dtype=torch.bool))
        delta = batch.domain[tag, q_left] - batch.domain[tag, batch.x_idx[tag, left]]
        infinite = torch.full_like(batch.objective, float("inf"))
        scores.append(matrix.exact_scores(batch, tag, left, None, delta, infinite,
                                          rows, options, Counters()))

    (dense_max, dense_l2), (sparse_max, sparse_l2) = scores
    assert torch.allclose(dense_max, sparse_max, rtol=1e-4, atol=1e-5)
    assert torch.allclose(dense_l2, sparse_l2, rtol=1e-3, atol=1e-4)


def test_l2_nonincrease_refuses_a_power_it_is_not_defined_at():
    """That policy is a constraint on the L2 norm, not a tie-break refinement."""
    A, domain, target = system(n_constraints=20, n_variables=8)
    with pytest.raises(ValueError, match="tiebreak_power=2"):
        minviol.solve(A, target, target, domain=domain, init="zero",
                      options=Options(acceptance=LINF_L2_NONINCREASE,
                                      tiebreak_power=4.0),
                      budget=Budget(seconds=0.2))


def test_a_raised_power_still_returns_a_legal_point():
    """Hard constraints 1 and 3, at a power the defaults never take."""
    A, domain, target = system(n_constraints=60, n_variables=16, seed=5)
    result = minviol.solve(A, target, target, domain=domain, init="zero",
                           options=Options(prune_min_constraints=0,
                                           tiebreak_power=6.0),
                           budget=Budget(seconds=10.0, max_iterations=8))
    assert torch.isin(torch.as_tensor(result.x), domain).all()
    recomputed = (torch.as_tensor(result.x, dtype=A.dtype) @ A.T - target).abs().max()
    assert recomputed == pytest.approx(result.max_violation, rel=1e-4, abs=1e-5)
