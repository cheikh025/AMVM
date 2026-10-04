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


def tiebreak_terms(v, power, scale=None):
    """Per-constraint terms of the tie-break quantity, summed by the caller.

    The tie-break is ``sum_i (v_i / scale) ** power`` over *every* constraint.
    Two properties earn their keep:

    **It is a sum over a fixed set**, which is what makes it a potential function
    and so what makes a plateau walk terminate. Summing over the worst
    constraints instead -- a set recomputed from the current point -- was measured
    and cycles; see experiment 7.

    **It is additive over constraints**, so the sparse backend can carry the
    incumbent's value and correct it only on the constraints a candidate touches.
    That holds for any power, not just two.

    ``power`` interpolates between the two natural choices. At 2 it is the sum of
    squares. As it grows the sum is dominated by the largest violations, so the
    ordering approaches the lexicographic one on the sorted violation vector,
    which is the quantity a minimax objective actually wants -- without the sort,
    and without moving the set. What bounds it in practice is float resolution,
    not cost: see experiment 8.

    ``scale`` divides the violations first. It must be a constant of the solve,
    not a function of the point, or the ordering it induces is a different
    potential function on every step. Its only job is to keep a high power inside
    the dynamic range of the accumulator.
    """
    if power == 2.0 and scale is None:
        return v.square()            # the incumbent path, bit for bit
    if scale is not None:
        v = v / scale
    return v.pow(power)


def check_policy(policy: str, is_equality: bool, tiebreak_power: float = 2.0) -> None:
    """Raise if ``policy`` is not usable on this problem."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown acceptance policy: {policy!r}; expected one of {POLICIES}")
    if tiebreak_power <= 0:
        raise ValueError(f"tiebreak_power must be positive, got {tiebreak_power!r}")
    if policy == LINF_L2_NONINCREASE and tiebreak_power != 2.0:
        raise ValueError(
            f"Acceptance policy {policy!r} constrains an accepted move not to raise the "
            f"sum of squares, so it is defined at tiebreak_power=2; got {tiebreak_power!r}. "
            "Raising the power is a tie-break refinement, not an L2 constraint -- the two "
            "happen to share a cache. Use 'linf_l2_tiebreak' to vary the power."
        )
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
