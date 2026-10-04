"""The compound two-variable path, checked against a from-scratch recomputation.

A compound move sets two variables to two levels with independent steps. Unlike a
swap those steps do not factor into one scaled column difference, so both backends
need a separate expression for it, and both need checking against brute force
rather than against each other.
"""
import torch

from minviol import engine, violation
from minviol.backends import DenseMatrix, SparseMatrix
from minviol.counters import Counters
from minviol.options import Options
from minviol.problem import Batch


def _instance(seed=0, m=400, n=20, levels=4, instances=2):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, n, generator=g)
    A[A.abs() < 1.2] = 0
    domain = torch.linspace(-1, 1, levels).repeat(instances, 1)
    x0 = torch.randint(0, levels, (instances, n), generator=g)
    planted = torch.randint(0, levels, (instances, n), generator=g)
    target = A @ torch.gather(domain, 1, planted).T
    return A, domain, x0, target - 0.3, target + 0.3


def _check(matrix, A, domain, x0, lower, upper):
    batch = Batch(matrix, x0, domain, lower, upper)
    options = Options(prune_min_constraints=0,
                      n_filters=max(128, getattr(matrix, "max_nnz", 100) + 1))
    rows = engine.screening_rows(batch, options.n_filters)

    g = torch.Generator().manual_seed(1)
    count, levels = 200, domain.shape[1]
    tag = torch.randint(0, x0.shape[0], (count,), generator=g)
    left = torch.randint(0, A.shape[1], (count,), generator=g)
    right = torch.randint(0, A.shape[1], (count,), generator=g)
    keep = left != right
    tag, left, right = tag[keep], left[keep], right[keep]
    q_left = torch.randint(0, levels, (len(tag),), generator=g)
    q_right = torch.randint(0, levels, (len(tag),), generator=g)
    delta = batch.domain[tag, q_left] - batch.domain[tag, batch.x_idx[tag, left]]
    delta_right = batch.domain[tag, q_right] - batch.domain[tag, batch.x_idx[tag, right]]

    maxima, squares = matrix.exact_scores(batch, tag, left, right, delta,
                                          batch.objective, rows, options, Counters(),
                                          delta_right=delta_right)
    for c in range(len(tag)):
        r = int(tag[c])
        x = batch.x_idx[r].clone()
        x[left[c]], x[right[c]] = q_left[c], q_right[c]
        v = violation(A @ domain[r][x], lower[:, r], upper[:, r])
        assert abs(float(v.max()) - float(maxima[c])) < 1e-4, f"candidate {c} maximum"
        assert abs(float(v.square().sum()) - float(squares[c])) < 1e-2 * max(
            1.0, float(v.square().sum())), f"candidate {c} squares"
    return len(tag)


def test_dense_compound_scores_match_recomputation():
    A, domain, x0, lower, upper = _instance()
    assert _check(DenseMatrix(A), A, domain, x0, lower, upper) > 100


def test_sparse_compound_scores_match_recomputation():
    A, domain, x0, lower, upper = _instance()
    assert _check(SparseMatrix.from_dense(A), A, domain, x0, lower, upper) > 100


def test_a_compound_move_can_leave_a_point_no_single_move_can():
    """The premise of the move: a pair improves where every single candidate fails."""
    import time
    # x1 + x2 = 2 wants both variables raised; 3(x1 - x2) = 0 punishes raising
    # either one alone by more than the first constraint gains. From (0,0) the
    # maximum violation is 2; every single move takes it to 3 or 6; moving both
    # to level 1 takes it to 0.
    A = torch.tensor([[1.0, 1.0], [3.0, -3.0]])
    domain = torch.tensor([[0.0, 1.0, 2.0]])
    bounds = torch.tensor([[2.0], [0.0]])
    batch = Batch(DenseMatrix(A), torch.zeros(1, 2, dtype=torch.long), domain,
                  bounds, bounds, acceptance="linf_l2_tiebreak")
    options = Options(prune_min_constraints=0, swap_moves=False, compound_moves=True,
                      acceptance="linf_l2_tiebreak", n_filters=2)
    active = torch.ones(1, dtype=torch.bool)

    engine.local_search(batch, active, Options(**{**options.__dict__,
                                                 "compound_moves": False}),
                        Counters(), deadline=time.time() + 30)
    single_best = float(batch.objective)
    assert single_best == 2.0, ("the point was meant to be a strict local optimum "
                                f"at 2.0, got {single_best}")
    engine.compound_pass(batch, active, engine.screening_rows(batch, 2), options,
                         Counters())
    assert float(batch.objective) < single_best, (
        "no compound move improved a point built so that only a pair can")
    assert float(batch.objective) == 0.0
