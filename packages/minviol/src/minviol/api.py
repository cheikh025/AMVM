"""Public entry points.

``solve`` answers one question: find a point ``x`` drawn from a discrete domain
such that ``lower <= A x <= upper``, or get as close as the budget allows. The
quantity minimized is the largest violation, so an objective of zero means every
constraint holds.

``solve_batch`` runs many instances that share one matrix in a single tensor
program. Instances are independent; batching them is worthwhile because one
small instance cannot fill a device on its own.
"""

import time
from dataclasses import dataclass, field, replace

import torch

from . import engine, initialize
from . import violation as viol
from .backends import DenseMatrix, SparseMatrix
from .counters import Counters
from .options import Budget, Options
from .problem import Batch


@dataclass
class Result:
    """What one instance's solve produced."""

    x: torch.Tensor           # physical values, guaranteed members of the domain
    x_index: torch.Tensor     # the same point as level indices
    max_violation: float      # unscaled: "does my constraint hold"
    objective: float          # scaled quantity the search minimized
    feasible: bool
    wall_time: float
    iterations: int
    counters: dict = field(default_factory=dict)

    def __repr__(self):
        state = "feasible" if self.feasible else f"max_violation={self.max_violation:.6g}"
        return (f"Result({state}, objective={self.objective:.6g}, "
                f"{self.iterations} iterations in {self.wall_time:.3f}s)")


def _as_matrix(A):
    """Accept a dense tensor, a torch sparse tensor, or an already-built backend.

    The backend follows what the caller handed over rather than a density
    heuristic: a guess that silently densified a matrix the caller deliberately
    kept sparse would be a surprising way to run out of memory.
    """
    if hasattr(A, "matvec"):
        return A
    if isinstance(A, torch.Tensor) and A.layout != torch.strided:
        return SparseMatrix.from_torch_sparse(A)
    A = torch.as_tensor(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D (constraints, variables), got shape {tuple(A.shape)}")
    return DenseMatrix(A)


def _default_filters(matrix):
    """How many constraints to keep in the screening set.

    For the sparse backend this is not a tuning knob. It sets the set the
    exactness argument quantifies over, and a candidate whose column covers all
    of it falls back to a bound instead of an exact score. Keeping it above the
    widest column makes that unreachable.
    """
    if getattr(matrix, "kind", "dense") == "sparse":
        return max(128, matrix.max_nnz + 1)
    return 100


def _as_bound(value, name, n_constraints, n_instances, device, dtype, default):
    """Normalize a bound to ``(constraints, instances)`` without copying when shared.

    ``expand`` gives a stride-0 view, so bounds shared across instances cost one
    vector rather than one per instance -- which matters, because there can be
    hundreds of thousands of constraints.
    """
    if value is None:
        return torch.full((n_constraints, 1), default, device=device,
                          dtype=dtype).expand(n_constraints, n_instances)
    t = torch.as_tensor(value, device=device, dtype=dtype)
    if t.ndim == 1:
        if t.shape[0] != n_constraints:
            raise ValueError(f"{name} has length {t.shape[0]}, expected {n_constraints}")
        return t[:, None].expand(n_constraints, n_instances)
    if t.ndim == 2:
        if t.shape == (n_instances, n_constraints):
            return t.T
        if t.shape == (n_constraints, n_instances):
            return t
        raise ValueError(f"{name} has shape {tuple(t.shape)}, expected "
                         f"({n_instances}, {n_constraints})")
    raise ValueError(f"{name} must be 1-D or 2-D, got {t.ndim}-D")


def _as_domain(domain, n_instances, device, dtype):
    d = torch.as_tensor(domain, device=device, dtype=dtype)
    if d.ndim == 1:
        d = d[None, :].expand(n_instances, d.shape[0])
    elif d.ndim != 2 or d.shape[0] != n_instances:
        raise ValueError(f"domain has shape {tuple(d.shape)}, expected "
                         f"({n_instances}, levels) or (levels,)")
    if d.shape[1] < 2:
        raise ValueError("domain needs at least two levels")
    if not bool((d.diff(dim=1) > 0).all()):
        raise ValueError("domain must be sorted strictly increasing")
    if not bool(torch.isfinite(d).all()):
        raise ValueError("domain contains non-finite levels")
    return d.contiguous()


def solve_batch(A, lower=None, upper=None, *, domain, init="lstsq_round", x0=None,
                budget=None, fixed=None, row_scale=None, options=None,
                n_instances=None):
    """Solve a batch of instances sharing the matrix ``A``. Returns a list of Results."""
    budget = budget or Budget()
    options = options or Options()
    matrix = _as_matrix(A)
    device, dtype = matrix.device, matrix.dtype
    n_constraints, n_variables = matrix.shape

    if n_instances is None:
        if x0 is not None:
            n_instances = torch.as_tensor(x0).reshape(-1, n_variables).shape[0]
        else:
            probe = torch.as_tensor(lower if lower is not None else upper)
            n_instances = probe.shape[0] if probe.ndim == 2 else 1

    lower_t = _as_bound(lower, "lower", n_constraints, n_instances, device, dtype,
                        float("-inf"))
    upper_t = _as_bound(upper, "upper", n_constraints, n_instances, device, dtype,
                        float("inf"))
    viol.check_bounds(lower_t, upper_t)
    viol.check_policy(options.acceptance, viol.is_equality(lower_t, upper_t),
                      options.tiebreak_power)

    domain_t = _as_domain(domain, n_instances, device, dtype)
    scale_t = None
    if row_scale is not None:
        scale_t = _as_bound(row_scale, "row_scale", n_constraints, n_instances, device,
                            dtype, 1.0)
        if bool((scale_t < 0).any()):
            raise ValueError("row_scale must be non-negative")

    if x0 is not None:
        x0 = torch.as_tensor(x0, device=device).reshape(n_instances, n_variables)
    x_idx = initialize.build(init, matrix, lower_t, upper_t, domain_t, n_variables, x0,
                             budget_mb=options.memory_budget_mb)
    if x_idx.shape != (n_instances, n_variables):
        raise ValueError(f"starting point has shape {tuple(x_idx.shape)}, expected "
                         f"({n_instances}, {n_variables})")
    if bool(((x_idx < 0) | (x_idx >= domain_t.shape[1])).any()):
        raise ValueError("starting point has level indices outside the domain")

    fixed_t = None
    if fixed is not None:
        fixed_t = torch.as_tensor(fixed, device=device, dtype=torch.bool)
        fixed_t = fixed_t.expand(n_instances, n_variables) if fixed_t.ndim == 1 else fixed_t

    if options.n_filters is None:
        options = replace(options, n_filters=_default_filters(matrix))

    counters = Counters()
    batch = Batch(matrix, x_idx, domain_t, lower_t, upper_t, row_scale=scale_t,
                  fixed_mask=fixed_t, acceptance=options.acceptance,
                  tiebreak_power=options.tiebreak_power, seed=options.seed)

    finite = torch.isfinite(lower_t)
    scale = float(lower_t[finite].abs().max()) if bool(finite.any()) else 0.0
    finite_upper = torch.isfinite(upper_t)
    if bool(finite_upper.any()):
        scale = max(scale, float(upper_t[finite_upper].abs().max()))
    feasibility_tol = budget.resolve_feasibility_tol(dtype, scale, n_constraints)

    started = time.time()
    iterations = _run(batch, budget, options, counters, started, feasibility_tol)
    elapsed = time.time() - started

    batch.x_idx = batch.best_x_idx.clone()
    batch.refresh()
    max_violation = batch.max_violation()
    physical = batch.physical(batch.x_idx)
    snapshot = counters.snapshot()

    return [Result(x=physical[r], x_index=batch.x_idx[r],
                   max_violation=float(max_violation[r]),
                   objective=float(batch.objective[r]),
                   feasible=bool(max_violation[r] <= feasibility_tol),
                   wall_time=elapsed, iterations=iterations, counters=snapshot)
            for r in range(n_instances)]


def _run(batch, budget, options, counters, started, feasibility_tol):
    """Perturb, descend, keep the best per instance, until the budget runs out."""
    device = batch.device
    active = torch.ones(batch.n_instances, dtype=torch.bool, device=device)
    deadline = started + budget.seconds

    # How many kicks in a row have failed to beat the incumbent, per instance.
    # Drives the kick width when escalation is on.
    stall = torch.zeros(batch.n_instances, dtype=torch.long, device=device)

    def kick_counts():
        """Per-instance kick width, doubling with each run of failures."""
        if not options.kick_escalation:
            return None
        base = max(1, int(round(options.destroy_rate * batch.n_variables)))
        ceiling = max(base, int(options.kick_max_rate * batch.n_variables))
        steps = (stall // max(1, options.kick_patience)).clamp(max=16)
        return (base * torch.pow(2, steps)).clamp(max=ceiling)

    def settle():
        """Record improvements and freeze instances that have reached feasibility."""
        better = batch.objective < batch.best_objective
        stall.copy_(torch.where(better, torch.zeros_like(stall), stall + 1))
        batch.best_x_idx[better] = batch.x_idx[better]
        batch.best_objective = torch.where(better, batch.objective, batch.best_objective)
        if budget.stop_when_feasible:
            done = batch.max_violation() <= feasibility_tol
            if bool(done.any()):
                # A feasible point answers the question this solver was asked.
                # Freezing rather than breaking lets the rest of the batch run on.
                batch.best_x_idx[done] = torch.where(done[:, None], batch.x_idx,
                                                     batch.best_x_idx)[done]
                batch.best_objective = torch.where(done, batch.objective,
                                                   batch.best_objective)
                active[done] = False
        return bool(active.any())

    engine.local_search(batch, active, options, counters, deadline=deadline)
    alive = settle()

    iterations = 0
    while alive and time.time() < deadline:
        if budget.max_iterations is not None and iterations >= budget.max_iterations:
            break
        iterations += 1
        batch.x_idx = batch.best_x_idx.clone()
        engine.perturb(batch, active, options.destroy_rate, options.perturbation,
                       kick_counts())
        engine.local_search(batch, active, options, counters, deadline=deadline)
        alive = settle()
        counters.bump("iterations")
        if iterations % options.refresh_every == 0:
            batch.x_idx = batch.best_x_idx.clone()
            batch.refresh()
    return iterations


def solve(A, lower=None, upper=None, *, domain, init="lstsq_round", x0=None,
          budget=None, fixed=None, row_scale=None, options=None):
    """Find a point satisfying ``lower <= A x <= upper`` with ``x`` drawn from ``domain``.

    Returns a single :class:`Result`. GPU parallelism here comes from evaluating
    candidates against constraints, not from running instances side by side, so a
    single large instance uses the device just as well as a batch does.
    """
    if x0 is not None:
        x0 = torch.as_tensor(x0).reshape(1, -1)
    return solve_batch(A, lower, upper, domain=domain, init=init, x0=x0, budget=budget,
                       fixed=fixed, row_scale=row_scale, options=options,
                       n_instances=1)[0]
