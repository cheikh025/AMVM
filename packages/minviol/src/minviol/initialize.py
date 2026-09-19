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


def fit_rows(matrix, rows, n_variables, budget_mb=256, seed=9101):
    """Choose which constraints to fit against, within a memory budget.

    The fit needs a dense block, and a sparse matrix has no dense block to lend:
    asking for every bounded constraint would materialize the whole matrix, which
    is precisely what the caller avoided by handing over a sparse one. At 100,000
    constraints and 4,096 variables that is 1.6 GB, spent on a starting point.

    So a sparse matrix is fitted against a sample. A least-squares fit does not
    need every constraint to point in the right direction, and the search is what
    refines it. Dense matrices are left alone: they already hold the block.
    """
    if getattr(matrix, "kind", "dense") != "sparse":
        return rows
    element_size = torch.empty(0, dtype=matrix.dtype).element_size()
    affordable = max(1, (budget_mb * 1024 * 1024) // max(1, n_variables * element_size))
    wanted = min(len(rows), max(64, min(16 * n_variables, affordable)))
    if wanted >= len(rows):
        return rows
    # Sampled rather than strided: constraints often arrive in a structured order
    # (an angle at a time, a block at a time), and a stride can draw the whole
    # sample from one structure.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    picked = torch.randperm(len(rows), generator=generator)[:wanted]
    return rows[picked.to(rows.device)]


def lstsq_round_start(matrix, lower, upper, domain, n_variables, budget_mb=256):
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
        rows = fit_rows(matrix, rows, n_variables, budget_mb)
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


def build(init, matrix, lower, upper, domain, n_variables, x0=None, budget_mb=256):
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
        return lstsq_round_start(matrix, lower, upper, domain, n_variables, budget_mb)
    raise ValueError(f"Unknown init {init!r}; expected 'zero', 'given' or 'lstsq_round'")
