"""The objective itself: violation, the bound checks, and the slack equivalence."""
import pytest
import torch

from minviol import violation
from minviol.violation import check_bounds, check_policy, is_equality


def test_equality_bounds_reproduce_the_absolute_residual():
    y = torch.tensor([-2.0, 0.5, 3.0])
    b = torch.tensor([0.0, 1.0, 1.0])
    assert torch.equal(violation(y, b, b), (y - b).abs())


def test_one_sided_bounds_are_zero_where_the_constraint_holds():
    y = torch.tensor([-5.0, 0.0, 2.0, 7.0])
    upper = torch.full((4,), 2.0)
    lower = torch.full((4,), -float("inf"))
    assert torch.equal(violation(y, lower, upper), torch.tensor([0.0, 0.0, 0.0, 5.0]))


def test_range_rows_measure_distance_to_the_nearer_bound():
    y = torch.tensor([-3.0, 0.0, 4.0])
    lower, upper = torch.full((3,), -1.0), torch.full((3,), 1.0)
    assert torch.equal(violation(y, lower, upper), torch.tensor([2.0, 0.0, 3.0]))


def test_violation_equals_the_optimal_slack_formulation():
    """The design rests on eliminating the slack analytically; check it against search.

    For ``a.x <= u`` the slack formulation minimizes ``|a.x + s - u|`` over
    ``s >= 0``. Brute-force that minimization and compare with the clamp.
    """
    torch.manual_seed(3)
    y = torch.randn(200) * 4
    upper = torch.randn(200)
    slacks = torch.linspace(0, 12, 4001)[None, :]
    by_search = (y[:, None] + slacks - upper[:, None]).abs().min(dim=1).values
    by_clamp = violation(y, torch.full_like(y, -float("inf")), upper)
    assert torch.allclose(by_search, by_clamp, atol=12 / 4000)


def test_lower_above_upper_is_rejected_as_infeasible_not_solved():
    lower = torch.tensor([0.0, 5.0, 0.0])
    upper = torch.tensor([1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="lower > upper"):
        check_bounds(lower, upper)


def test_l2_policies_are_refused_on_inequalities():
    """The squared positive part is zero wherever a one-sided row holds, so the
    tie-break would stop discriminating rather than merely weaken."""
    check_policy("linf_l2_tiebreak", is_equality=True)
    with pytest.raises(ValueError, match="only defined when every constraint"):
        check_policy("linf_l2_tiebreak", is_equality=False)
    with pytest.raises(ValueError, match="only defined when every constraint"):
        check_policy("linf_l2_nonincrease", is_equality=False)
    check_policy("linf", is_equality=False)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown acceptance policy"):
        check_policy("greedy", is_equality=True)


def test_is_equality_detects_mixed_systems():
    """One non-equality row is enough to disqualify the L2 policies.

    The comparison is exact, so a widening below the dtype's resolution is not a
    widening at all: 3.0 + 1e-9 is 3.0 in float32. That is the right behaviour --
    a bound the arithmetic cannot distinguish from equality is an equality -- but
    the test has to use a difference the dtype can hold.
    """
    b = torch.tensor([1.0, 2.0, 3.0])
    assert is_equality(b, b)
    assert not is_equality(b, b + torch.tensor([0.0, 0.0, 1e-3]))
    assert is_equality(b, b + torch.tensor([0.0, 0.0, 1e-9]))
