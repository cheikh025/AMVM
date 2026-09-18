"""Constraint violation, and the acceptance policies defined on it.

The solver minimizes the largest violation of ``lower <= A x <= upper``. For one
constraint that is

    v = max(0, lower - y, y - upper)      where y = (A x)

which is zero exactly when the constraint holds. Setting ``lower == upper == b``
gives ``|y - b|``, the two-sided residual the code this was lifted from
minimized, so equality problems are a special case rather than a separate path.

Inequalities are handled here, not by adding slack variables. For ``a.x <= u``
the slack formulation minimizes ``|a.x + s - u|`` subject to ``s >= 0``, and for
fixed ``x`` the best slack is ``s* = max(0, u - a.x)``, which makes that term
exactly ``max(0, a.x - u)``. The slack has a closed-form optimum, so it is
eliminated analytically instead of searched over. Searching it would put one
extra variable per constraint into the dimension that costs the most.
"""

import torch

LINF = "linf"
LINF_L2_TIEBREAK = "linf_l2_tiebreak"
LINF_L2_NONINCREASE = "linf_l2_nonincrease"

POLICIES = (LINF, LINF_L2_TIEBREAK, LINF_L2_NONINCREASE)

# `linf_l2_nonincrease` constrains an accepted move to not raise the sum of
# squares. On a one-sided row that is currently satisfied the positive part is
# exactly zero, so that constraint is measuring almost nothing once most rows
# hold, and it only ever removes moves. It stays equality-only.
#
# `linf_l2_tiebreak` is the opposite: it *adds* moves, by letting a candidate
# that leaves the maximum where it was win on the sum of squares. That is how a
# search crosses the plateau a maximum objective creates, and refusing it on
# inequalities was measured to cost more than it saved -- see experiment 2.
EQUALITY_ONLY_POLICIES = (LINF_L2_NONINCREASE,)


def violation(y, lower, upper):
    """Return ``max(0, lower - y, y - upper)`` elementwise.

    ``y``, ``lower`` and ``upper`` broadcast against each other, so this serves
    both a single point and a whole candidate block.
    """
    return torch.maximum((lower - y).clamp_min(0), (y - upper).clamp_min(0))


def check_policy(policy: str, is_equality: bool) -> None:
    """Raise if ``policy`` is not usable on this problem."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown acceptance policy: {policy!r}; expected one of {POLICIES}")
    if policy in EQUALITY_ONLY_POLICIES and not is_equality:
        raise ValueError(
            f"Acceptance policy {policy!r} is only defined when every constraint is an "
            "equality (lower == upper). On a one-sided constraint the squared positive "
            "part is zero wherever the constraint holds, so this policy would forbid "
            "moves on the strength of a quantity that is measuring nothing. Use 'linf' "
            "or 'linf_l2_tiebreak'."
        )


def check_bounds(lower, upper) -> None:
    """Raise if any constraint is unsatisfiable as written."""
    if lower.shape != upper.shape:
        raise ValueError(f"lower and upper must have the same shape, got {tuple(lower.shape)} "
                         f"and {tuple(upper.shape)}")
    bad = lower > upper
    if bool(bad.any()):
        first = int(torch.nonzero(bad.reshape(-1), as_tuple=True)[0][0])
        raise ValueError(
            f"lower > upper on {int(bad.sum())} constraint(s), first at flat index {first}. "
            "No point can satisfy such a row, so the problem is infeasible as written "
            "rather than hard."
        )
    if bool(torch.isnan(lower).any() or torch.isnan(upper).any()):
        raise ValueError("lower/upper contain NaN")


def is_equality(lower, upper, atol: float = 0.0) -> bool:
    """True when every constraint is an equality, so the L2 policies apply."""
    if atol:
        return bool(((upper - lower).abs() <= atol).all())
    return bool((lower == upper).all())
