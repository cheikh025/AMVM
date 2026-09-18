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

# The two L2 policies are only defined when every row is an equality. On a
# one-sided row that is currently satisfied the positive part is exactly zero,
# so the sum of squares stops discriminating between candidates as soon as most
# rows are satisfied, and the tie-break degenerates to noise on the few violated
# ones. Under lower == upper it is a full signal, which is why the quantization
# work never met this.
EQUALITY_ONLY_POLICIES = (LINF_L2_TIEBREAK, LINF_L2_NONINCREASE)


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
            "part is zero wherever the constraint holds, so the L2 term stops "
            "discriminating. Use 'linf'."
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
