"""Starting points, and the one-sided case that used to fall through.

The default start rounds a least-squares fit onto the domain. Where it aims each
constraint is not a detail: a rule that only handled two-sided rows left a pure
inequality system with no usable rows at all, and quietly fell back to the cold
start. On a 100,000-constraint system that was the difference between finding a
feasible point in 0.05s and not finding one in 20s.
"""
import pytest
import torch

import minviol
from minviol import Budget
from minviol.initialize import least_squares_target


def test_two_sided_constraints_aim_at_the_middle():
    lower = torch.tensor([0.0, -4.0])
    upper = torch.tensor([2.0, 6.0])
    target, usable = least_squares_target(lower, upper)
    assert torch.equal(target, torch.tensor([1.0, 1.0]))
    assert bool(usable.all())


def test_one_sided_constraints_aim_at_their_own_bound():
    """``a.x == u`` satisfies ``a.x <= u``, so the bound is the nearest happy point."""
    inf = float("inf")
    lower = torch.tensor([-inf, 3.0])
    upper = torch.tensor([5.0, inf])
    target, usable = least_squares_target(lower, upper)
    assert torch.equal(target, torch.tensor([5.0, 3.0]))
    assert bool(usable.all())


def test_wholly_unbounded_constraints_are_dropped():
    inf = float("inf")
    lower = torch.tensor([-inf, 0.0])
    upper = torch.tensor([inf, 1.0])
    _, usable = least_squares_target(lower, upper)
    assert usable.tolist() == [False, True]


def test_inequality_system_gets_a_real_warm_start():
    """The regression: a pure ``A x <= u`` system must not start from zero."""
    torch.manual_seed(0)
    A = torch.randn(4000, 64)
    domain = torch.arange(4.0)
    planted = domain[torch.randint(0, 4, (64,))]
    upper = A @ planted + 1.0

    warm = minviol.solve(A, None, upper, domain=domain, init="lstsq_round",
                         budget=Budget(seconds=0.0))
    cold = minviol.solve(A, None, upper, domain=domain, init="zero",
                         budget=Budget(seconds=0.0))

    assert warm.max_violation < cold.max_violation, (
        "the least-squares start is no better than the cold one on an inequality "
        "system, which is what the one-sided target rule exists to prevent")


def test_zero_start_picks_the_level_nearest_zero():
    A = torch.eye(3)
    domain = torch.tensor([-5.0, -0.5, 4.0])
    result = minviol.solve(A, domain=domain, init="zero", budget=Budget(seconds=0.0))
    assert torch.equal(result.x, torch.full((3,), -0.5))


def test_a_problem_with_no_finite_bound_still_starts_somewhere():
    A = torch.eye(3)
    domain = torch.tensor([0.0, 1.0])
    result = minviol.solve(A, domain=domain, init="lstsq_round",
                           budget=Budget(seconds=0.0))
    assert bool(torch.isin(result.x, domain).all())


def test_sparse_fit_block_stays_inside_the_memory_budget():
    """The default start must not densify a matrix the caller kept sparse.

    Fitting against every bounded constraint would materialize the whole matrix,
    which is the failure the sparse backend exists to avoid -- and it would
    happen before the search ran a single iteration.
    """
    from minviol.backends import SparseMatrix
    from minviol.initialize import fit_rows

    torch.manual_seed(0)
    n_constraints, n_variables = 20_000, 512
    dense = torch.randn(n_constraints, n_variables)
    dense[torch.rand_like(dense) > 0.01] = 0
    matrix = SparseMatrix.from_dense(dense)

    rows = torch.arange(n_constraints)
    budget_mb = 4
    chosen = fit_rows(matrix, rows, n_variables, budget_mb=budget_mb)

    block_bytes = len(chosen) * n_variables * 4
    assert block_bytes <= budget_mb * 1024 * 1024, (
        f"fit block is {block_bytes / 2**20:.1f} MiB against a {budget_mb} MiB budget")
    assert len(chosen) < n_constraints, "nothing was sampled"
    assert len(chosen) >= 64, "sampled too few constraints to fit against"


def test_dense_matrices_fit_against_every_row():
    """A dense matrix already holds the block, so nothing is sampled from it."""
    from minviol.backends import DenseMatrix
    from minviol.initialize import fit_rows

    matrix = DenseMatrix(torch.randn(5000, 64))
    rows = torch.arange(5000)
    assert torch.equal(fit_rows(matrix, rows, 64, budget_mb=1), rows)


def test_sparse_warm_start_still_beats_the_cold_one_after_sampling():
    """Sampling must not cost the warm start its advantage."""
    torch.manual_seed(1)
    n_constraints, n_variables = 20_000, 128
    dense = torch.randn(n_constraints, n_variables)
    dense[torch.rand_like(dense) > 0.02] = 0
    domain = torch.arange(4.0)
    planted = domain[torch.randint(0, 4, (n_variables,))]
    upper = dense @ planted + 1.0
    sparse = dense.to_sparse_coo()

    warm = minviol.solve(sparse, None, upper, domain=domain, init="lstsq_round",
                         budget=Budget(seconds=0.0),
                         options=minviol.Options(memory_budget_mb=4))
    cold = minviol.solve(sparse, None, upper, domain=domain, init="zero",
                         budget=Budget(seconds=0.0))
    assert warm.max_violation < cold.max_violation
