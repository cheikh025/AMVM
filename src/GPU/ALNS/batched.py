"""Tier B: solve every row of a weight matrix in one batched tensor program.

All rows of a matrix share the same activation matrix and are independent, so a
row solve is a small program repeated thousands of times. Running them one at a
time leaves the device mostly idle: the per-row work is not large enough to fill
it, and every row pays the full Python and launch cost on its own. This module
holds the rows together instead, as ``(rows, variables)`` weights and
``(samples, rows)`` residuals, and evaluates one candidate list that carries a
row tag, so a swap pass over the whole matrix is a handful of large kernels.

What this is not: it does not use the ``alns`` package. That package drives one
instance per Python loop and owns operator selection and acceptance, which is
exactly the structure batching has to remove. The loop here is deliberately
small (destroy, repair, batched swap, hill-climbing acceptance) and matches what
the quantization path actually runs. Tomography and the FIR design keep using
the package.

Rows finish at different times, so a row that can no longer improve is masked
out of the candidate build and stops costing anything, while the rest continue.
"""

import time

import torch

from ALNS import counters, tuning

EPSILON = 0.001


class BatchedRows:
    """A batch of independent row problems over one shared activation matrix.

    Residuals are stored as ``(samples, rows)`` so that a candidate column drawn
    for row ``r`` lines up with the ``(samples, candidates)`` block produced by
    gathering activation columns, with no transpose in the inner loop.
    """

    def __init__(self, inputs, weights, levels, B_k, acceptance_policy="linf_l2_nonincrease",
                 fixed_mask=None, seed=9101):
        self.inputs = inputs
        self.device = inputs.device
        self.dtype = inputs.dtype
        self.weights = weights.long().clone()
        self.levels = levels
        self.B_k = B_k
        self.acceptance_policy = acceptance_policy
        self.n_rows, self.n_variables = self.weights.shape
        self.n_levels = levels.shape[1]
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.fixed_mask = (torch.zeros_like(self.weights, dtype=torch.bool)
                           if fixed_mask is None else fixed_mask.clone())

        self.residual = self.recompute_residual(self.weights)
        self.objective = self.residual.abs().max(dim=0).values
        self.l2 = self.residual.square().sum(dim=0)
        self.best_weights = self.weights.clone()
        self.best_objective = self.objective.clone()

    def quantized(self, weights):
        """Return the physical weights of ``weights`` as ``(rows, variables)``."""
        return torch.gather(self.levels, 1, weights)

    def recompute_residual(self, weights):
        """Return ``(samples, rows)`` residuals from scratch, without cached state."""
        return self.inputs @ self.quantized(weights).T - self.B_k.T

    def refresh(self, rows=None):
        """Recompute residual and objective caches from scratch.

        Incremental updates accumulate float error over thousands of accepted
        moves, so the driver calls this periodically and at the end. Without it
        the reported objective slowly drifts from the true one.
        """
        self.residual = self.recompute_residual(self.weights)
        self.objective = self.residual.abs().max(dim=0).values
        self.l2 = self.residual.square().sum(dim=0)

    def apply_swaps(self, rows, left, right, q1, q2):
        """Swap levels ``q1`` and ``q2`` between variables ``left`` and ``right``."""
        if not len(rows):
            return
        # Index the levels of the rows being changed. Gathering positionally would
        # silently read the first len(rows) rows' domains instead, which matters as
        # soon as rows carry different level values or only some rows improve.
        delta = self.levels[rows, q2] - self.levels[rows, q1]
        self.residual[:, rows] += delta[None, :] * (self.inputs[:, left] - self.inputs[:, right])
        self.weights[rows, left] = q2
        self.weights[rows, right] = q1
        self.objective = self.residual.abs().max(dim=0).values
        self.l2 = self.residual.square().sum(dim=0)


def candidate_lists(batch, q1, q2, active):
    """Build the flat candidate list for one level pair across every active row.

    Returns ``(tag, left, right)``: for each candidate, which row it belongs to
    and which two variables it swaps. The cross product of the two level groups
    is formed on the device with one host transfer for the group sizes, so the
    number of Python statements does not grow with the number of candidates.
    """
    device = batch.device
    assignable = ~batch.fixed_mask
    at_q1 = (batch.weights == q1) & assignable
    at_q2 = (batch.weights == q2) & assignable
    at_q1 &= active[:, None]
    at_q2 &= active[:, None]

    counts_i = at_q1.sum(dim=1)
    counts_j = at_q2.sum(dim=1)
    pairs = counts_i * counts_j
    total = int(pairs.sum().item())  # one transfer, to size the candidate list
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    # Members of each level group, row by row, in increasing variable order.
    order_i = torch.argsort(at_q1.int(), dim=1, descending=True, stable=True)
    order_j = torch.argsort(at_q2.int(), dim=1, descending=True, stable=True)

    tag = torch.repeat_interleave(torch.arange(batch.n_rows, device=device), pairs)
    offsets = torch.cumsum(pairs, 0) - pairs
    within = torch.arange(total, device=device) - offsets[tag]
    local_i = within // counts_j[tag]
    local_j = within % counts_j[tag]
    left = order_i[tag, local_i]
    right = order_j[tag, local_j]
    return tag, left, right


def screen(batch, tag, left, right, delta, bound, screen_rows, tile):
    """Return the candidates that stay under ``bound`` on each row's worst samples."""
    survivors = []
    for start in range(0, len(tag), tile):
        piece = slice(start, start + tile)
        rows = screen_rows[tag[piece]].T                      # (screen, candidates)
        residual = (batch.residual[rows, tag[piece][None, :]]
                    + delta[piece][None, :] * (batch.inputs[rows, left[piece][None, :]]
                                               - batch.inputs[rows, right[piece][None, :]]))
        keep = (residual.abs() <= bound[tag[piece]][None, :]).all(dim=0)
        survivors.append(keep)
    return torch.cat(survivors) if survivors else torch.zeros(0, dtype=torch.bool,
                                                              device=batch.device)


def exact_scores(batch, tag, left, right, delta, prune_bound):
    """Exact infinity norm and sum of squares per candidate, pruning as it goes.

    Same argument as the single-row path: a running maximum only grows, so a
    candidate already above its row's best can never be chosen and its remaining
    samples are dead work.
    """
    device, n_samples = batch.device, batch.residual.shape[0]
    live = torch.arange(len(tag), device=device)
    maxima = torch.zeros(len(tag), dtype=batch.dtype, device=device)
    squares = torch.zeros_like(maxima)

    budget = max(1, tuning.SWAP_MEMORY_BUDGET_MB * 1024 * 1024
                 // (batch.inputs.element_size() * tuning.LIVE_INTERMEDIATES))
    start = 0
    stage = min(tuning.PRUNE_FIRST_STAGE, n_samples)
    while start < n_samples and len(live):
        rows = slice(start, start + stage)
        residual = (batch.residual[rows][:, tag[live]]
                    + delta[live][None, :] * (batch.inputs[rows][:, left[live]]
                                              - batch.inputs[rows][:, right[live]]))
        maxima[live] = torch.maximum(maxima[live], residual.abs().max(dim=0).values)
        squares[live] += residual.square().sum(dim=0)
        counters.bump("batched_row_candidate_products",
                      (min(start + stage, n_samples) - start) * len(live))
        start += stage
        if start < n_samples:
            live = live[maxima[live] <= prune_bound[tag[live]]]
            if len(live):
                stage = int(min(max(tuning.PRUNE_FIRST_STAGE,
                                    min(stage * tuning.PRUNE_STAGE_GROWTH,
                                        budget // len(live))),
                                n_samples - start))
    return maxima, squares


def best_per_row(batch, tag, maxima, squares, admissible):
    """Pick one winning candidate per row: least maximum, then least squares.

    Three segment reductions rather than a scan, so the choice costs the same
    whether a row offers ten candidates or a million. Ties are settled by the
    smallest candidate position, which keeps the choice reproducible.
    """
    infinity = torch.tensor(float("inf"), device=batch.device, dtype=batch.dtype)
    keyed_linf = torch.where(admissible, maxima, infinity)
    row_linf = torch.full((batch.n_rows,), float("inf"), device=batch.device,
                          dtype=batch.dtype)
    row_linf.scatter_reduce_(0, tag, keyed_linf, reduce="amin", include_self=True)

    # Rows with no admissible candidate keep an infinite best. Comparing that to
    # the candidates' own infinite keys would call all of them ties and hand the
    # row a swap it must not take, so admissibility gates the tie test.
    ties = admissible & (keyed_linf == row_linf[tag])
    keyed_l2 = torch.where(ties, squares, infinity)
    row_l2 = torch.full_like(row_linf, float("inf"))
    row_l2.scatter_reduce_(0, tag, keyed_l2, reduce="amin", include_self=True)

    winners = ties & (squares == row_l2[tag]) & torch.isfinite(row_linf[tag])
    positions = torch.where(winners, torch.arange(len(tag), device=batch.device),
                            torch.full_like(tag, len(tag)))
    row_position = torch.full((batch.n_rows,), len(tag), device=batch.device,
                              dtype=torch.long)
    row_position.scatter_reduce_(0, tag, positions, reduce="amin", include_self=True)
    return row_linf, row_position


def swap_pass(batch, active, screen_rows, candidate_tile=4096):
    """One batched swap pass. Returns the rows that improved."""
    device = batch.device
    if batch.acceptance_policy == "linf_l2_tiebreak":
        bound = batch.objective + EPSILON
    else:
        bound = batch.objective - EPSILON

    improved = torch.zeros(batch.n_rows, dtype=torch.bool, device=device)
    for q1 in range(batch.n_levels - 1):
        q2 = q1 + 1
        tag, left, right = candidate_lists(batch, q1, q2, active & ~improved)
        if not len(tag):
            continue
        delta = batch.levels[tag, q2] - batch.levels[tag, q1]
        alive = delta != 0
        tag, left, right, delta = tag[alive], left[alive], right[alive], delta[alive]
        if not len(tag):
            continue

        keep = screen(batch, tag, left, right, delta, bound, screen_rows, candidate_tile)
        counters.bump("batched_screened", len(tag))
        tag, left, right, delta = tag[keep], left[keep], right[keep], delta[keep]
        counters.bump("batched_survivors", len(tag))
        if not len(tag):
            continue

        maxima, squares = exact_scores(batch, tag, left, right, delta, bound)
        if batch.acceptance_policy == "linf_l2_tiebreak":
            admissible = ((maxima < batch.objective[tag] - EPSILON)
                          | ((maxima - batch.objective[tag]).abs() <= EPSILON)
                          & (squares < batch.l2[tag]))
        else:
            admissible = maxima < batch.objective[tag] - EPSILON
        if batch.acceptance_policy == "linf_l2_nonincrease":
            admissible = admissible & (squares <= batch.l2[tag])

        _, position = best_per_row(batch, tag, maxima, squares, admissible)
        chosen = position[position < len(tag)]
        rows = torch.nonzero(position < len(tag), as_tuple=True)[0]
        if not len(rows):
            continue
        batch.apply_swaps(rows, left[chosen], right[chosen],
                          torch.full_like(rows, q1), torch.full_like(rows, q2))
        improved[rows] = True
        counters.bump("batched_swaps_applied", len(rows))
    return improved


def local_search(batch, active, n_filters=100, candidate_tile=4096, max_passes=1000):
    """Run batched swap passes until no active row improves."""
    working = active.clone()
    for _ in range(max_passes):
        screen_rows = batch.residual.abs().T.topk(min(n_filters,
                                                      batch.residual.shape[0]), dim=1).indices
        improved = swap_pass(batch, working, screen_rows, candidate_tile)
        counters.bump("batched_passes")
        working = working & improved
        if not bool(working.any()):
            break
    return batch


def destroy_and_repair(batch, active, destroy_rate=0.005):
    """Perturb each active row: move a random subset of its variables by one level."""
    assignable = (~batch.fixed_mask) & active[:, None]
    count = max(1, int(round(destroy_rate * batch.n_variables)))
    scores = torch.rand(batch.weights.shape, generator=batch.generator, device=batch.device)
    scores = torch.where(assignable, scores, torch.full_like(scores, -1.0))
    chosen = scores.topk(count, dim=1).indices

    steps = torch.randint(-1, 2, chosen.shape, generator=batch.generator, device=batch.device)
    current = torch.gather(batch.weights, 1, chosen)
    proposed = (current + steps).clamp(0, batch.n_levels - 1)
    proposed = torch.where(active[:, None], proposed, current)
    batch.weights.scatter_(1, chosen, proposed)
    batch.residual = batch.recompute_residual(batch.weights)
    batch.objective = batch.residual.abs().max(dim=0).values
    batch.l2 = batch.residual.square().sum(dim=0)


def solve(inputs, weights, levels, B_k, seconds, acceptance_policy="linf_l2_nonincrease",
          fixed_mask=None, seed=9101, destroy_rate=0.005, n_filters=100,
          candidate_tile=4096, refresh_every=25):
    """Run the batched loop for a wall-clock budget and return the best weights.

    Hill climbing per row: a perturbation that leaves a row worse than its best is
    rolled back for that row alone, so rows neither wait for each other nor share
    an acceptance decision.
    """
    batch = BatchedRows(inputs, weights, levels, B_k, acceptance_policy, fixed_mask, seed)
    active = torch.ones(batch.n_rows, dtype=torch.bool, device=batch.device)
    local_search(batch, active, n_filters, candidate_tile)
    improved = batch.objective < batch.best_objective
    batch.best_weights[improved] = batch.weights[improved]
    batch.best_objective = torch.minimum(batch.objective, batch.best_objective)

    deadline = time.time() + seconds
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        batch.weights = batch.best_weights.clone()
        destroy_and_repair(batch, active, destroy_rate)
        local_search(batch, active, n_filters, candidate_tile)

        improved = batch.objective < batch.best_objective
        batch.best_weights[improved] = batch.weights[improved]
        batch.best_objective = torch.where(improved, batch.objective, batch.best_objective)
        counters.bump("batched_iterations")
        if iteration % refresh_every == 0:
            batch.weights = batch.best_weights.clone()
            batch.refresh()

    batch.weights = batch.best_weights.clone()
    batch.refresh()
    return batch.best_weights, batch.objective
