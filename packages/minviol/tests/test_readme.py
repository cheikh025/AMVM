"""The README's opening example has to actually run.

A published example that does not execute is worse than no example: it is the
first thing a reader tries and the first thing that makes them distrust the rest.
"""
import torch

import minviol


def test_readme_opening_example_runs():
    torch.manual_seed(0)
    domain = torch.arange(0, 8.0)
    A = torch.randn(500, 32)
    planted = domain[torch.randint(0, 8, (32,))]
    target = A @ planted
    lower, upper = target - 0.5, target + 0.5

    result = minviol.solve(
        A,
        lower, upper,
        domain=domain,
        budget=minviol.Budget(seconds=10.0),
    )

    assert isinstance(result.feasible, bool)
    assert bool(torch.isin(result.x, domain).all())
    assert result.max_violation >= 0.0
    if result.feasible:
        assert result.max_violation == 0.0


def test_readme_batch_example_runs():
    torch.manual_seed(1)
    domain = torch.arange(0, 4.0)
    A = torch.randn(200, 16)
    planted = domain[torch.randint(0, 4, (3, 16))]
    targets = (A @ planted.T).T
    lowers, uppers = targets - 0.5, targets + 0.5

    results = minviol.solve_batch(A, lowers, uppers, domain=domain,
                                  budget=minviol.Budget(seconds=2.0))
    assert len(results) == 3
    assert all(bool(torch.isin(r.x, domain).all()) for r in results)


def test_every_documented_constraint_shape_is_accepted():
    """The four rows in the README's table, in one system."""
    A = torch.eye(4)
    domain = torch.tensor([0.0, 1.0, 2.0])
    inf = float("inf")
    lower = torch.tensor([-inf, 1.0, 2.0, 0.0])
    upper = torch.tensor([1.0, inf, 2.0, 2.0])

    result = minviol.solve(A, lower, upper, domain=domain,
                           budget=minviol.Budget(seconds=1.0))
    assert result.feasible, f"left {result.max_violation} on a satisfiable system"
