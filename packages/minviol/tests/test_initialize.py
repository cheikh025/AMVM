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
