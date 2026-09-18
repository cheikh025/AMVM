"""Input handling: what the API accepts, what it refuses, and what it refuses to guess."""
import pytest
import torch

import minviol
from minviol import Budget, Options


@pytest.fixture
def tiny():
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    domain = torch.tensor([0.0, 1.0, 2.0])
    return A, domain


def test_shared_bounds_are_not_materialized_per_instance(tiny):
    """A bound shared across instances must cost one vector, not one per instance.

    There can be hundreds of thousands of constraints, so broadcasting this
    wrong is the difference between a vector and a matrix.
    """
    from minviol.api import _as_bound
    shared = _as_bound(torch.zeros(3), "lower", 3, 64, torch.device("cpu"),
                       torch.float32, 0.0)
    assert shared.shape == (3, 64)
    assert shared.stride(1) == 0, "shared bounds were copied per instance"


def test_unsorted_domain_is_refused(tiny):
    A, _ = tiny
    with pytest.raises(ValueError, match="sorted strictly increasing"):
        minviol.solve(A, domain=torch.tensor([1.0, 0.0, 2.0]), budget=Budget(seconds=0.0))


def test_domain_with_repeated_levels_is_refused(tiny):
    A, _ = tiny
    with pytest.raises(ValueError, match="sorted strictly increasing"):
        minviol.solve(A, domain=torch.tensor([0.0, 0.0, 1.0]), budget=Budget(seconds=0.0))


def test_single_level_domain_is_refused(tiny):
    A, _ = tiny
    with pytest.raises(ValueError, match="at least two levels"):
        minviol.solve(A, domain=torch.tensor([0.0]), budget=Budget(seconds=0.0))


def test_contradictory_bounds_are_refused(tiny):
    A, domain = tiny
    lower = torch.tensor([0.0, 5.0, 0.0])
    upper = torch.tensor([1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="lower > upper"):
        minviol.solve(A, lower, upper, domain=domain, budget=Budget(seconds=0.0))


def test_l2_policy_on_inequalities_is_refused(tiny):
    A, domain = tiny
    with pytest.raises(ValueError, match="only defined when every constraint"):
        minviol.solve(A, None, torch.zeros(3), domain=domain,
                      options=Options(acceptance="linf_l2_nonincrease"),
                      budget=Budget(seconds=0.0))


def test_given_init_without_a_point_is_refused(tiny):
    A, domain = tiny
    with pytest.raises(ValueError, match="requires x0"):
        minviol.solve(A, domain=domain, init="given", budget=Budget(seconds=0.0))


def test_a_starting_point_with_no_init_to_use_it_is_refused(tiny):
    """Silently ignoring x0 would hand back an answer to a different question."""
    A, domain = tiny
    with pytest.raises(ValueError, match="pass init='given'"):
        minviol.solve(A, domain=domain, init="zero", x0=torch.zeros(2, dtype=torch.long),
                      budget=Budget(seconds=0.0))


def test_out_of_domain_starting_point_is_refused(tiny):
    A, domain = tiny
    with pytest.raises(ValueError, match="outside the domain"):
        minviol.solve(A, domain=domain, init="given",
                      x0=torch.tensor([0, 7]), budget=Budget(seconds=0.0))


def test_wrong_length_bound_is_refused(tiny):
    A, domain = tiny
    with pytest.raises(ValueError, match="expected 3"):
        minviol.solve(A, torch.zeros(4), torch.zeros(4), domain=domain,
                      budget=Budget(seconds=0.0))


def test_negative_row_scale_is_refused(tiny):
    A, domain = tiny
    with pytest.raises(ValueError, match="non-negative"):
        minviol.solve(A, torch.zeros(3), torch.zeros(3), domain=domain,
                      row_scale=torch.tensor([1.0, -1.0, 1.0]), budget=Budget(seconds=0.0))


def test_an_already_feasible_start_is_recognized_immediately(tiny):
    """Nothing to search for; the answer is the starting point."""
    A, domain = tiny
    x_star = torch.tensor([1.0, 2.0])
    target = A @ x_star
    result = minviol.solve(A, target, target, domain=domain, init="given",
                           x0=torch.tensor([1, 2]), budget=Budget(seconds=5.0))
    assert result.feasible
    assert result.iterations == 0
    assert result.wall_time < 1.0
    assert torch.equal(result.x, x_star)


def test_unbounded_problem_has_nothing_to_violate(tiny):
    """With no finite bound every point is feasible, and the solver should say so."""
    A, domain = tiny
    result = minviol.solve(A, domain=domain, budget=Budget(seconds=1.0))
    assert result.feasible
    assert result.max_violation == 0.0


def test_a_zero_column_variable_cannot_change_anything():
    """A variable absent from every constraint must not be reported as improving."""
    A = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    domain = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([2.0, 4.0])
    result = minviol.solve(A, target, target, domain=domain, init="given",
                           x0=torch.tensor([0, 0]), budget=Budget(seconds=0.5))
    assert result.feasible
    assert float(result.x[0]) == 2.0
