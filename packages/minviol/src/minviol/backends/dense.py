"""Dense constraint matrix.

Every candidate touches every constraint, so the kernels here are large regular
blocks: the whole point is to keep the device busy and the host out of the loop.
Screening cuts the candidate list against each instance's worst constraints
before the exact stage pays for all of them.
"""

import torch

from .. import violation as viol
from ..options import LIVE_INTERMEDIATES, prune_slack


class DenseMatrix:
    """``(constraints, variables)`` dense matrix."""

    kind = "dense"

    def __init__(self, A):
        self.A = A
        self.device = A.device
        self.dtype = A.dtype
        self.shape = tuple(A.shape)

    def matvec(self, x_phys):
        """``(instances, variables)`` point -> ``(constraints, instances)`` values."""
        return self.A @ x_phys.T

    def columns(self, cols):
        """``A[:, cols]`` as ``(constraints, len(cols))``."""
        return self.A[:, cols]

    def apply_delta(self, y, instances, left, right, delta):
        """Add each instance's move to its column of ``y``, in place."""
        column = self.A[:, left]
        if right is not None:
            column = column - self.A[:, right]
        y[:, instances] += delta[None, :] * column

    # -- candidate scoring -------------------------------------------------

    def _candidate_y(self, base_y, delta_sel, left_sel, right_sel, row_index):
        """Values of ``A x`` for a candidate block, from already-selected candidates.

        ``right_sel is None`` is a single-variable move; the two-variable case is
        the swap. Keeping both in one expression is what lets the single-variable
        pass reuse this backend unchanged.
        """
        column = self.A[row_index, left_sel]
        if right_sel is not None:
            column = column - self.A[row_index, right_sel]
        return base_y + delta_sel * column

    def screen(self, batch, tag, left, right, delta, bound, screen_rows, tile):
        """Return the candidates staying under ``bound`` on their instance's worst rows."""
        survivors = []
        for start in range(0, len(tag), tile):
            piece = slice(start, start + tile)
            tags = tag[piece]
            rows = screen_rows.indices[tags].T               # (screen, candidates)
            cols = tags[None, :]
            base = batch.y[rows, cols]
            y = self._candidate_y(base, delta[piece][None, :], left[piece],
                                  None if right is None else right[piece], rows)
            v = viol.violation(y, batch.lower[rows, cols], batch.upper[rows, cols])
            if batch.row_scale is not None:
                v = v * batch.row_scale[rows, cols]
            survivors.append((v <= bound[tags][None, :]).all(dim=0))
        return (torch.cat(survivors) if survivors
                else torch.zeros(0, dtype=torch.bool, device=self.device))

    def exact_scores(self, batch, tag, left, right, delta, prune_bound, screen_rows,
                     options, counters):
        """Exact maximum violation and sum of squares per candidate, pruning as it goes.

        A running maximum only grows, so a candidate already above its instance's
        best can never be chosen and its remaining constraints are dead work.
        """
        device = self.device
        n_constraints = batch.n_constraints
        live = torch.arange(len(tag), device=device)
        maxima = torch.zeros(len(tag), dtype=self.dtype, device=device)
        squares = torch.zeros_like(maxima)

        budget = max(1, options.memory_budget_mb * 1024 * 1024
                     // (self.A.element_size() * LIVE_INTERMEDIATES))
        start = 0
        staged = n_constraints >= options.prune_min_constraints
        stage = min(options.prune_first_stage, n_constraints) if staged else n_constraints
        while start < n_constraints and len(live):
            rows = slice(start, start + stage)
            base = batch.y[rows][:, tag[live]]
            y = self._candidate_y(base, delta[live][None, :], left[live],
                                  None if right is None else right[live], rows)
            v = viol.violation(y, batch.lower[rows][:, tag[live]],
                               batch.upper[rows][:, tag[live]])
            if batch.row_scale is not None:
                v = v * batch.row_scale[rows][:, tag[live]]
            maxima[live] = torch.maximum(maxima[live], v.max(dim=0).values)
            squares[live] += v.square().sum(dim=0)
            counters.bump("constraint_candidate_products",
                          (min(start + stage, n_constraints) - start) * len(live))
            start += stage
            if start < n_constraints:
                live = live[maxima[live] <= prune_slack(prune_bound[tag[live]])]
                if len(live):
                    stage = int(min(max(options.prune_first_stage,
                                        min(stage * options.prune_stage_growth,
                                            budget // len(live))),
                                    n_constraints - start))
        return maxima, squares

    def dense_rows(self, rows):
        """``A[rows, :]`` as a dense block, for the least-squares start."""
        return self.A[rows]
