"""The sparse backend must be the dense one, computed differently.

It reaches the same scores by reading only the constraints a candidate touches
plus a bounded set of the worst ones. Everything here is aimed at the places
that shortcut could be wrong: the bound it rests on, the constraints touched by
both halves of a swap, and the incremental sums.
"""
import pytest
import torch

import minviol
from minviol import Budget, Options, engine
from minviol.backends import DenseMatrix, SparseMatrix
from minviol.counters import Counters
from minviol.problem import Batch


def sparse_instance(device, m=400, n=28, levels=4, instances=3, cutoff=1.5, seed=1):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, n, generator=g)
    A[A.abs() < cutoff] = 0
    A = A.to(device)
    domain = torch.linspace(-1, 1, levels).repeat(instances, 1).to(device)
    x0 = torch.randint(0, levels, (instances, n), generator=g).to(device)
    planted = torch.randint(0, levels, (instances, n), generator=g).to(device)
    target = A @ torch.gather(domain, 1, planted).T
    return A, domain, x0, target


def paired(device, policy="linf", slack=0.2, **kwargs):
    A, domain, x0, target = sparse_instance(device, **kwargs)
    lower, upper = (target, target) if slack == 0 else (target - slack, target + slack)
    sparse = SparseMatrix.from_dense(A)
    options = Options(acceptance=policy, prune_min_constraints=0,
                      n_filters=max(128, sparse.max_nnz + 1))
    dense_batch = Batch(DenseMatrix(A), x0, domain, lower, upper, acceptance=policy)
    sparse_batch = Batch(sparse, x0, domain, lower, upper, acceptance=policy)
    return A, dense_batch, sparse_batch, options


def test_construction_matches_the_dense_matrix(device):
    A, dense_batch, sparse_batch, _ = paired(device)
    assert torch.allclose(dense_batch.y, sparse_batch.y, atol=1e-5)
    assert torch.allclose(dense_batch.objective, sparse_batch.objective, atol=1e-5)
    assert sparse_batch.matrix.nnz == int((A != 0).sum())


def test_duplicate_entries_are_refused(device):
    """A duplicate is invisible to A @ x but applied twice incrementally.

    That failure mode is slow drift, not a crash, so it has to be caught at
    construction rather than found later in a quality number.
    """
    indices = torch.tensor([[0, 1, 0], [0, 1, 0]], device=device)
    values = torch.tensor([1.0, 2.0, 3.0], device=device)
    with pytest.raises(ValueError, match="duplicate"):
        SparseMatrix(indices, values, (2, 2))


def test_indices_outside_the_shape_are_refused(device):
    indices = torch.tensor([[0, 5], [0, 1]], device=device)
    values = torch.tensor([1.0, 2.0], device=device)
    with pytest.raises(ValueError, match="outside the declared shape"):
        SparseMatrix(indices, values, (2, 2))


def test_membership_test_agrees_with_the_dense_matrix(device):
    """``contains`` is a binary search; check it against the matrix it indexes."""
    A, _, sparse_batch, _ = paired(device)
    matrix = sparse_batch.matrix
    m, n = A.shape
    rows = torch.arange(m, device=device).repeat_interleave(n)
    cols = torch.arange(n, device=device).repeat(m)
    assert torch.equal(matrix.contains(cols, rows), (A[rows, cols] != 0))


@pytest.mark.parametrize("policy", ["linf", "linf_l2_nonincrease"])
def test_candidate_scores_match_dense_elementwise(device, policy):
    """The kernel oracle: same candidates, same scores, independent of the search."""
    _, dense_batch, sparse_batch, options = paired(device, policy=policy, slack=0)
    active = torch.ones(dense_batch.n_instances, dtype=torch.bool, device=device)
    rows_d = engine.screening_rows(dense_batch, options.n_filters)
    rows_s = engine.screening_rows(sparse_batch, options.n_filters)
    counters = Counters()

    tag, left, level = engine.candidate_moves(dense_batch, active)
    delta = (dense_batch.domain[tag, level]
             - dense_batch.domain[tag, dense_batch.x_idx[tag, left]])
    for right, ltag, lleft, ldelta in [(None, tag, left, delta)]:
        dm, ds = dense_batch.matrix.exact_scores(
            dense_batch, ltag, lleft, right, ldelta, dense_batch.objective, rows_d,
            options, counters)
        sm, ss = sparse_batch.matrix.exact_scores(
            sparse_batch, ltag, lleft, right, ldelta, sparse_batch.objective, rows_s,
            options, counters)
        # The maximum is a max, so it does not depend on accumulation order and
        # must agree exactly. The sum of squares is accumulated differently by the
        # two backends, so it agrees to float32 rounding only.
        assert torch.equal(dm, sm)
        assert torch.allclose(ds, ss, rtol=1e-4, atol=1e-2)
    assert counters["untouched_max_inexact"] == 0


def test_swap_scores_match_dense_including_shared_constraints(device):
    """A swap's two columns can touch the same constraint; both halves must land."""
    _, dense_batch, sparse_batch, options = paired(device, slack=0, cutoff=0.5)
    active = torch.ones(dense_batch.n_instances, dtype=torch.bool, device=device)
    rows_d = engine.screening_rows(dense_batch, options.n_filters)
    rows_s = engine.screening_rows(sparse_batch, options.n_filters)
    counters = Counters()

    found = 0
    for q1 in range(dense_batch.n_levels - 1):
        tag, left, right = engine.candidate_swaps(dense_batch, q1, q1 + 1, active)
        if not len(tag):
            continue
        found += len(tag)
        delta = dense_batch.domain[tag, q1 + 1] - dense_batch.domain[tag, q1]
        dm, _ = dense_batch.matrix.exact_scores(dense_batch, tag, left, right, delta,
                                                dense_batch.objective, rows_d, options,
                                                counters)
        sm, _ = sparse_batch.matrix.exact_scores(sparse_batch, tag, left, right, delta,
                                                 sparse_batch.objective, rows_s, options,
                                                 counters)
        assert torch.equal(dm, sm), f"swap scores diverged at level pair {q1}"
    assert found, "no swap candidates were generated, so nothing was tested"


def test_trajectory_matches_dense_step_for_step(device):
    """Under pure linf the winning score is a max, so the two must agree exactly."""
    _, dense_batch, sparse_batch, options = paired(device)
    active = torch.ones(dense_batch.n_instances, dtype=torch.bool, device=device)
    for step in range(15):
        engine.local_search(dense_batch, active, options, Counters())
        engine.local_search(sparse_batch, active, options, Counters())
        assert torch.equal(dense_batch.x_idx, sparse_batch.x_idx), \
            f"trajectories diverged after descent {step}"
        engine.perturb(dense_batch, active, 0.05)
        engine.perturb(sparse_batch, active, 0.05)
        assert torch.equal(dense_batch.x_idx, sparse_batch.x_idx)


def test_a_column_covering_every_screened_constraint_stays_sound(device):
    """Force the bound that a wide enough screening set makes unreachable.

    With K below the widest column support, a candidate can touch every screened
    constraint and leave nothing to read the untouched maximum from. The score
    then falls back to tau, which must still be an upper bound -- accepting on an
    upper bound is sound, accepting on an underestimate is not.
    """
    A = torch.tensor([[3.0, 1.0], [2.0, 1.0], [1.0, 0.0], [0.5, 0.0]], device=device)
    domain = torch.tensor([[0.0, 1.0, 2.0]], device=device)
    x0 = torch.zeros(1, 2, dtype=torch.long, device=device)
    bounds = torch.zeros(4, 1, device=device)
    sparse_batch = Batch(SparseMatrix.from_dense(A), x0, domain, bounds, bounds)
    dense_batch = Batch(DenseMatrix(A), x0, domain, bounds, bounds)

    # K = 2, and column 0 touches every constraint, so T \ S is empty.
    options = Options(n_filters=2, prune_min_constraints=0)
    rows_s = engine.screening_rows(sparse_batch, 2)
    rows_d = engine.screening_rows(dense_batch, 2)
    counters = Counters()

    tag = torch.zeros(2, dtype=torch.long, device=device)
    left = torch.zeros(2, dtype=torch.long, device=device)
    level = torch.tensor([1, 2], device=device)
    delta = domain[0, level] - domain[0, 0]

    exact, _ = dense_batch.matrix.exact_scores(dense_batch, tag, left, None, delta,
                                               dense_batch.objective, rows_d, options,
                                               counters)
    bounded, _ = sparse_batch.matrix.exact_scores(sparse_batch, tag, left, None, delta,
                                                  sparse_batch.objective, rows_s, options,
                                                  counters)
    assert counters["untouched_max_inexact"] == 2, "the fallback branch never ran"
    assert torch.all(bounded >= exact - 1e-6), "the fallback underestimated a score"


def test_default_screening_set_makes_the_fallback_unreachable(device):
    """The shipped default keeps K above the widest column, so the bound is exact."""
    _, _, sparse_batch, _ = paired(device, cutoff=0.2)
    matrix = sparse_batch.matrix
    from minviol.api import _default_filters
    assert _default_filters(matrix) > matrix.max_nnz


def test_solve_accepts_a_torch_sparse_tensor(device):
    A, domain, x0, target = sparse_instance(device, instances=1)
    result = minviol.solve(A.to_sparse_coo(), target[:, 0], target[:, 0],
                           domain=domain[0], init="given", x0=x0[0],
                           budget=Budget(seconds=1.0))
    recomputed = minviol.violation(A @ result.x, target[:, 0], target[:, 0]).max()
    assert result.max_violation == pytest.approx(float(recomputed), abs=1e-4)


def test_sparse_and_dense_solve_reach_the_same_answer(device):
    A, domain, x0, target = sparse_instance(device, instances=1)
    common = dict(domain=domain[0], init="given", x0=x0[0], budget=Budget(seconds=2.0))
    dense = minviol.solve(A, target[:, 0], target[:, 0], **common)
    sparse = minviol.solve(A.to_sparse_coo(), target[:, 0], target[:, 0], **common)
    assert torch.equal(dense.x_index, sparse.x_index)
