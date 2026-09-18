"""The search: candidate generation, acceptance, and the solve loop.

The loop is deliberately small -- perturb, descend, keep the best per instance.
Instances finish at different times, so one that can no longer improve is masked
out of candidate generation and stops costing anything while the rest continue.
An instance that reaches feasibility is frozen outright: the question was "find
a point that satisfies these constraints", and it has been answered.
"""

import time

import torch

from . import violation as viol


def candidate_swaps(batch, q1, q2, active):
    """Flat candidate list swapping levels ``q1`` and ``q2``, across active instances.

    Returns ``(tag, left, right)``: for each candidate, which instance it belongs
    to and which two variables it swaps. The cross product of the two level
    groups is formed on the device with one host transfer for the group sizes, so
    the number of Python statements does not grow with the number of candidates.
    """
    device = batch.device
    assignable = ~batch.fixed_mask & active[:, None]
    at_q1 = (batch.x_idx == q1) & assignable
    at_q2 = (batch.x_idx == q2) & assignable

    counts_i = at_q1.sum(dim=1)
    counts_j = at_q2.sum(dim=1)
    pairs = counts_i * counts_j
    total = int(pairs.sum().item())  # one transfer, to size the candidate list
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    # Members of each level group, instance by instance, in increasing variable order.
    order_i = torch.argsort(at_q1.int(), dim=1, descending=True, stable=True)
    order_j = torch.argsort(at_q2.int(), dim=1, descending=True, stable=True)

    tag = torch.repeat_interleave(torch.arange(batch.n_instances, device=device), pairs)
    offsets = torch.cumsum(pairs, 0) - pairs
    within = torch.arange(total, device=device) - offsets[tag]
    left = order_i[tag, within // counts_j[tag]]
    right = order_j[tag, within % counts_j[tag]]
    return tag, left, right


def candidate_moves(batch, active):
    """Flat candidate list of every ``(variable, level)`` move, across active instances.

    Unlike a swap, a single-variable move changes the multiset of assigned
    levels. That is why it exists: swaps alone cannot reach a point with a
    different level histogram, so from a cold start on a coarse domain a
    swap-only search has nothing to do.

    Every level is enumerated in one list rather than one level at a time. A
    swap's step is fixed by the level pair, which is why the swap pass can loop
    over pairs and let each one see a whole candidate list; a single-variable
    step differs per candidate regardless, so looping would buy nothing and would
    make the pass take the first improving level instead of the best one.
    """
    device = batch.device
    movable = ~batch.fixed_mask & active[:, None]
    index = torch.nonzero(movable, as_tuple=False)
    empty = torch.empty(0, dtype=torch.long, device=device)
    if not len(index):
        return empty, empty, empty

    n_levels = batch.n_levels
    tag = index[:, 0].repeat_interleave(n_levels)
    left = index[:, 1].repeat_interleave(n_levels)
    level = torch.arange(n_levels, device=device).repeat(len(index))

    moving = level != batch.x_idx[tag, left]
    return tag[moving], left[moving], level[moving]


def best_per_instance(batch, tag, maxima, squares, admissible):
    """Pick one winning candidate per instance: least maximum, then least squares.

    Three segment reductions rather than a scan, so the choice costs the same
    whether an instance offers ten candidates or a million. Ties are settled by
    the smallest candidate position, which keeps the choice reproducible.
    """
    device = batch.device
    infinity = torch.tensor(float("inf"), device=device, dtype=batch.dtype)
    keyed_linf = torch.where(admissible, maxima, infinity)
    inst_linf = torch.full((batch.n_instances,), float("inf"), device=device,
                           dtype=batch.dtype)
    inst_linf.scatter_reduce_(0, tag, keyed_linf, reduce="amin", include_self=True)

    # Instances with no admissible candidate keep an infinite best. Comparing that
    # to the candidates' own infinite keys would call all of them ties and hand the
    # instance a move it must not take, so admissibility gates the tie test.
    ties = admissible & (keyed_linf == inst_linf[tag])
    keyed_l2 = torch.where(ties, squares, infinity)
    inst_l2 = torch.full_like(inst_linf, float("inf"))
    inst_l2.scatter_reduce_(0, tag, keyed_l2, reduce="amin", include_self=True)

    winners = ties & (squares == inst_l2[tag]) & torch.isfinite(inst_linf[tag])
    positions = torch.where(winners, torch.arange(len(tag), device=device),
                            torch.full_like(tag, len(tag)))
    inst_position = torch.full((batch.n_instances,), len(tag), device=device,
                               dtype=torch.long)
    inst_position.scatter_reduce_(0, tag, positions, reduce="amin", include_self=True)
    return inst_linf, inst_position


def _admissibility(batch, tag, maxima, squares, eps):
    """Which candidates the acceptance policy would take."""
    incumbent = batch.objective[tag]
    step = eps[tag]
    if batch.acceptance == viol.LINF_L2_TIEBREAK:
        admissible = ((maxima < incumbent - step)
                      | ((maxima - incumbent).abs() <= step) & (squares < batch.l2[tag]))
    else:
        admissible = maxima < incumbent - step
    if batch.acceptance == viol.LINF_L2_NONINCREASE:
        admissible = admissible & (squares <= batch.l2[tag])
    return admissible


def _evaluate_and_apply(batch, tag, left, right, delta, q_left, q_right,
                        screen_rows, options, counters, improved):
    """Screen, score and apply the best candidate per instance. Returns nothing."""
    eps = options.improvement_epsilon(batch.objective)
    if not torch.is_tensor(eps):
        eps = torch.full_like(batch.objective, eps)
    bound = (batch.objective + eps if batch.acceptance == viol.LINF_L2_TIEBREAK
             else batch.objective - eps)

    keep = batch.matrix.screen(batch, tag, left, right, delta, bound, screen_rows,
                               options.candidate_tile)
    counters.bump("screened", len(tag))
    tag, left, delta = tag[keep], left[keep], delta[keep]
    right = None if right is None else right[keep]
    q_left = q_left[keep]
    q_right = None if q_right is None else q_right[keep]
    counters.bump("survivors", len(tag))
    if not len(tag):
        return

    maxima, squares = batch.matrix.exact_scores(batch, tag, left, right, delta, bound,
                                                options, counters)
    admissible = _admissibility(batch, tag, maxima, squares, eps)
    _, position = best_per_instance(batch, tag, maxima, squares, admissible)

    chosen = position[position < len(tag)]
    instances = torch.nonzero(position < len(tag), as_tuple=True)[0]
    if not len(instances):
        return
    batch.apply_moves(instances, left[chosen],
                      None if right is None else right[chosen],
                      q_left[chosen],
                      None if q_right is None else q_right[chosen])
    improved[instances] = True
    counters.bump("moves_applied", len(instances))


def swap_pass(batch, active, screen_rows, options, counters):
    """One batched swap pass. Returns the instances that improved."""
    improved = torch.zeros(batch.n_instances, dtype=torch.bool, device=batch.device)
    for q1 in range(batch.n_levels - 1):
        q2 = q1 + 1
        tag, left, right = candidate_swaps(batch, q1, q2, active & ~improved)
        if not len(tag):
            continue
        delta = batch.domain[tag, q2] - batch.domain[tag, q1]
        alive = delta != 0
        tag, left, right, delta = tag[alive], left[alive], right[alive], delta[alive]
        if not len(tag):
            continue
        _evaluate_and_apply(batch, tag, left, right, delta,
                            torch.full_like(tag, q2), torch.full_like(tag, q1),
                            screen_rows, options, counters, improved)
    return improved


def move_pass(batch, active, screen_rows, options, counters):
    """One batched single-variable pass. Returns the instances that improved.

    Every level is considered, not only the neighbouring ones. Holding the rest
    of the point fixed, each constraint's violation is a max of affine functions
    of the chosen value, so a candidate's score is convex along the sorted
    domain. A truncated window would therefore be a real restriction on the
    neighbourhood rather than a free one, and full enumeration is affordable
    while the domain is small -- which is the regime this solver is for.
    """
    improved = torch.zeros(batch.n_instances, dtype=torch.bool, device=batch.device)
    tag, left, level = candidate_moves(batch, active)
    if not len(tag):
        return improved

    delta = batch.domain[tag, level] - batch.domain[tag, batch.x_idx[tag, left]]
    alive = delta != 0
    tag, left, level, delta = tag[alive], left[alive], level[alive], delta[alive]
    if not len(tag):
        return improved

    _evaluate_and_apply(batch, tag, left, None, delta, level, None,
                        screen_rows, options, counters, improved)
    return improved


def screening_rows(batch, n_filters):
    """The worst constraints per instance, which every candidate is screened against.

    Keyed on the violation, not on the raw residual: with one-sided or asymmetric
    bounds the largest residuals are not the most violated constraints, and
    screening against the wrong set silently discards good candidates.
    """
    v = batch.violation_of(batch.y)
    k = min(n_filters, batch.n_constraints)
    return v.T.topk(k, dim=1).indices


def local_search(batch, active, options, counters, deadline=None, max_passes=1000):
    """Descend until no active instance improves.

    A pass is the smallest unit that leaves the batch consistent, so the deadline
    is checked between passes. Without it a single descent can run far past the
    budget, which makes an equal-time comparison meaningless.

    Single-variable moves run before swaps. Swaps preserve the multiset of
    assigned levels, so from a point with the wrong level histogram no sequence
    of swaps can reach a better one.
    """
    n_filters = options.n_filters or 100
    working = active.clone()
    for _ in range(max_passes):
        if deadline is not None and time.time() >= deadline:
            break
        rows = screening_rows(batch, n_filters)
        improved = torch.zeros_like(working)
        if options.single_variable_moves:
            improved |= move_pass(batch, working, rows, options, counters)
            rows = screening_rows(batch, n_filters)
        improved |= swap_pass(batch, working, rows, options, counters)
        counters.bump("passes")
        working = working & improved
        if not bool(working.any()):
            break
    return batch


def perturb(batch, active, destroy_rate):
    """Move a random subset of each active instance's variables by one level."""
    assignable = (~batch.fixed_mask) & active[:, None]
    count = max(1, int(round(destroy_rate * batch.n_variables)))
    scores = torch.rand(batch.x_idx.shape, generator=batch.generator, device=batch.device)
    scores = torch.where(assignable, scores, torch.full_like(scores, -1.0))
    chosen = scores.topk(count, dim=1).indices

    steps = torch.randint(-1, 2, chosen.shape, generator=batch.generator,
                          device=batch.device)
    current = torch.gather(batch.x_idx, 1, chosen)
    proposed = (current + steps).clamp(0, batch.n_levels - 1)
    proposed = torch.where(active[:, None], proposed, current)
    batch.x_idx.scatter_(1, chosen, proposed)
    batch.refresh()
