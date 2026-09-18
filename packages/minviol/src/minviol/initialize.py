"""Starting points.

Every application this engine was lifted from built its own starting point and
handed it in: quantization rounded the unquantized weights, tomography rounded a
filtered back-projection, filter design rounded its own least-squares fit. All
three are "round a target vector onto the domain", and the target is not
something a caller of ``solve(A, lower, upper, domain)`` has said anything
about. So the general API has to be able to produce one, or a plain call starts
from a cold point that the search is measurably bad at leaving.
"""

import torch


def nearest_level(domain, values):
    """Index of the closest level to each value. ``domain`` is (instances, levels)."""
    distance = (values[:, :, None] - domain[:, None, :]).abs()
    return distance.argmin(dim=2)


def zero_start(domain, n_variables):
    """Every variable at the level closest to zero."""
    target = torch.zeros(domain.shape[0], n_variables, dtype=domain.dtype,
                         device=domain.device)
    return nearest_level(domain, target)


def least_squares_target(lower, upper):
    """The value to aim each constraint at, and which constraints have one.

    A two-sided constraint is aimed at the middle of its interval. A one-sided one
    is aimed at its own bound: ``a.x == u`` satisfies ``a.x <= u``, so the
    boundary is the nearest point that the constraint is happy with, and anything
    the fit undershoots by is strictly feasible.

    Getting this wrong is not a small loss. Dropping one-sided constraints for
    want of a midpoint leaves a pure inequality system with no finite rows at
    all, so the fit degenerates to the zero start -- which is exactly the cold
    start this function exists to avoid.
    """
    has_lower, has_upper = torch.isfinite(lower), torch.isfinite(upper)
    both = has_lower & has_upper
    target = torch.where(both, 0.5 * (lower + upper),
                         torch.where(has_upper, upper,
                                     torch.where(has_lower, lower,
                                                 torch.zeros_like(lower))))
    return target, has_lower | has_upper


def lstsq_round_start(matrix, lower, upper, domain, n_variables):
    """Least-squares fit to the constraints' targets, rounded onto the domain.

    A constraint with no finite bound says nothing about where to look, so it is
    dropped. If that leaves nothing, this degenerates to the zero start, which is
    the honest answer for a problem with no bounds at all.
    """
    device = domain.device
    n_instances = domain.shape[0]
    targets, bounded = least_squares_target(lower, upper)

    starts = []
    for r in range(n_instances):
        rows = torch.nonzero(bounded[:, r], as_tuple=True)[0]
        if not len(rows):
            starts.append(zero_start(domain[r:r + 1], n_variables)[0])
            continue
        target = targets[rows, r]
        block = matrix.dense_rows(rows)
        # lstsq is not implemented on every backend (notably MPS), and this runs
        # once per solve, so a CPU round trip costs nothing worth defending.
        try:
            fit = torch.linalg.lstsq(block, target.unsqueeze(1)).solution[:, 0]
        except (RuntimeError, NotImplementedError):
            fit = torch.linalg.lstsq(block.cpu(), target.cpu().unsqueeze(1)
                                     ).solution[:, 0].to(device)
        fit = torch.nan_to_num(fit, nan=0.0, posinf=0.0, neginf=0.0)
        starts.append(nearest_level(domain[r:r + 1], fit.unsqueeze(0).to(domain.dtype))[0])
    return torch.stack(starts)


def build(init, matrix, lower, upper, domain, n_variables, x0=None):
    """Return the starting ``(instances, variables)`` level indices."""
    if init == "given":
        if x0 is None:
            raise ValueError("init='given' requires x0")
        return torch.as_tensor(x0, dtype=torch.long, device=domain.device)
    if x0 is not None:
        raise ValueError(f"x0 was supplied but init={init!r}; pass init='given' to use it")
    if init == "zero":
        return zero_start(domain, n_variables)
    if init == "lstsq_round":
        return lstsq_round_start(matrix, lower, upper, domain, n_variables)
    raise ValueError(f"Unknown init {init!r}; expected 'zero', 'given' or 'lstsq_round'")
