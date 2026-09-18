"""The batch of independent problems the search operates on.

All instances share one constraint matrix ``A`` and differ in their bounds,
their domain and their current point, which is what makes a batch worth holding
together: the per-instance work is too small to fill a device on its own, and
running instances one at a time pays the Python and launch cost for each.

Orientation: ``y``, ``lower`` and ``upper`` are ``(constraints, instances)`` so
that a candidate column drawn for instance ``r`` lines up with the
``(constraints, candidates)`` block produced by gathering matrix columns, with
no transpose in the inner loop. The public API takes the more natural
``(instances, constraints)`` and transposes once, here.
"""

import torch

from . import violation as viol


class Batch:
    """Current point, bounds and cached objective for a batch of instances."""

    def __init__(self, matrix, x_idx, domain, lower, upper, *, row_scale=None,
                 fixed_mask=None, acceptance=viol.LINF, seed=9101):
        self.matrix = matrix
        self.device = matrix.device
        self.dtype = matrix.dtype

        # x_idx holds indices into the domain, not physical values. The physical
        # point is domain[instance, x_idx], which is what the matrix multiplies.
        self.x_idx = x_idx.long().clone()
        self.domain = domain
        self.n_instances, self.n_variables = self.x_idx.shape
        self.n_levels = domain.shape[1]
        self.n_constraints = matrix.shape[0]

        self.lower = lower
        self.upper = upper
        self.row_scale = row_scale
        self.acceptance = acceptance

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.fixed_mask = (torch.zeros_like(self.x_idx, dtype=torch.bool)
                           if fixed_mask is None else fixed_mask.clone())

        self.y = self.recompute_y(self.x_idx)
        self._refresh_caches()
        self.best_x_idx = self.x_idx.clone()
        self.best_objective = self.objective.clone()

    # -- state ------------------------------------------------------------

    def physical(self, x_idx):
        """Return the physical values of ``x_idx`` as ``(instances, variables)``."""
        return torch.gather(self.domain, 1, x_idx)

    def recompute_y(self, x_idx):
        """Return ``(constraints, instances)`` values of ``A x``, from scratch."""
        return self.matrix.matvec(self.physical(x_idx))

    def violation_of(self, y):
        """Scaled violation of a ``(constraints, ...)`` block of ``A x`` values."""
        v = viol.violation(y, self.lower, self.upper)
        return v if self.row_scale is None else v * self.row_scale

    def _refresh_caches(self):
        v = self.violation_of(self.y)
        self.objective = v.max(dim=0).values
        self.l2 = v.square().sum(dim=0)

    def refresh(self):
        """Rebuild every cache from scratch.

        Incremental updates accumulate float error over thousands of accepted
        moves, so the driver calls this periodically and at the end. Without it
        the reported objective slowly drifts from the true one.
        """
        self.y = self.recompute_y(self.x_idx)
        self._refresh_caches()

    def max_violation(self):
        """Unscaled maximum violation per instance -- the feasibility question.

        ``objective`` is what the search minimizes and carries ``row_scale``;
        this is the raw answer to "does my constraint hold".
        """
        return viol.violation(self.y, self.lower, self.upper).max(dim=0).values

    # -- moves ------------------------------------------------------------

    def apply_compound(self, instances, left, right, q_left, q_right):
        """Set two variables to two levels at once, per listed instance.

        Unlike a swap the two steps are independent, so they cannot be folded into
        one scaled column difference. Both are read before either index is
        written, and the two are summed into a single residual update so that a
        constraint touched by both columns is updated once.
        """
        if not len(instances):
            return
        delta_left = (self.domain[instances, q_left]
                      - self.domain[instances, self.x_idx[instances, left]])
        delta_right = (self.domain[instances, q_right]
                       - self.domain[instances, self.x_idx[instances, right]])
        update = (delta_left[None, :] * self.matrix.columns(left)
                  + delta_right[None, :] * self.matrix.columns(right))
        self.y[:, instances] += update
        self.x_idx[instances, left] = q_left
        self.x_idx[instances, right] = q_right
        self._refresh_caches()

    def apply_moves(self, instances, left, right, q_left, q_right):
        """Apply one move per listed instance and update the caches.

        A move sets variable ``left`` to level ``q_left``; when ``right`` is not
        None it also sets variable ``right`` to ``q_right``, which is the swap
        the search uses most. Both shapes go through one code path so that the
        single-variable pass costs no extra kernel.
        """
        if not len(instances):
            return
        # Index the domain of the instances being changed. Gathering positionally
        # would silently read the first len(instances) domains instead, which
        # matters as soon as instances carry different level values or only some
        # of them improve.
        # Index the domain of the instances being changed. Gathering positionally
        # would silently read the first len(instances) domains instead, which
        # matters as soon as instances carry different level values or only some
        # of them improve.
        delta = (self.domain[instances, q_left]
                 - self.domain[instances, self.x_idx[instances, left]])
        # A swap moves the two variables by equal and opposite steps, so one delta
        # covers both, and the backend assembles the column difference before
        # scaling it. That is the expression the candidate was scored with:
        # applying an algebraically equal but differently grouped one would leave
        # the realized objective a unit in the last place away from the predicted
        # one, on every accepted move.
        self.matrix.apply_delta(self.y, instances, left, right, delta)
        self.x_idx[instances, left] = q_left
        if right is not None:
            self.x_idx[instances, right] = q_right

        self._refresh_caches()
