"""The search: candidate generation, acceptance, and the solve loop.

The loop is deliberately small -- perturb, descend, keep the best per instance.
Instances finish at different times, so one that can no longer improve is masked
out of candidate generation and stops costing anything while the rest continue.
An instance that reaches feasibility is frozen outright: the question was "find
a point that satisfies these constraints", and it has been answered.
"""

import time
from typing import NamedTuple

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


def _admissibility(batch, tag, maxima, squares, eps, reference):
    """Which candidates the acceptance policy would take.

    Ties are settled on the sum of squares over *every* constraint, and that
    "every" is load-bearing. It makes the quantity a potential function: a
    sideways move must strictly reduce it, so a walk across a plateau cannot
    return to a point it has left and must terminate. Settling ties on the worst
    constraints instead -- which is where the objective actually lives, and so
    looks like the better choice -- was measured and is not: that set is
    recomputed from the current point, so lowering the sum over it can raise
    constraints just outside it, which then enter it. The walk stops terminating.
    See experiment 7.
    """
    incumbent = batch.objective[tag]
    step = eps[tag]
    if batch.acceptance == viol.LINF_L2_TIEBREAK:
        admissible = ((maxima < incumbent - step)
                      | ((maxima - incumbent).abs() <= step) & (squares < reference[tag]))
    else:
        admissible = maxima < incumbent - step
    if batch.acceptance == viol.LINF_L2_NONINCREASE:
        admissible = admissible & (squares <= reference[tag])
    return admissible


def _evaluate_and_apply(batch, tag, left, right, delta, q_left, q_right,
                        screen_rows, options, counters, improved, delta_right=None,
                        compound=False):
    """Screen, score and apply the best candidate per instance. Returns nothing."""
    eps = options.improvement_epsilon(batch.objective)
    if not torch.is_tensor(eps):
        eps = torch.full_like(batch.objective, eps)
    bound = (batch.objective + eps if batch.acceptance == viol.LINF_L2_TIEBREAK
             else batch.objective - eps)

    if compound:
        # Screening assumes a swap's shared step, so it would score a compound
        # candidate as something it is not. The exact stage handles these.
        keep = torch.ones(len(tag), dtype=torch.bool, device=batch.device)
    else:
        keep = batch.matrix.screen(batch, tag, left, right, delta, bound, screen_rows,
                                   options.candidate_tile)
    counters.bump("screened", len(tag))
    tag, left, delta = tag[keep], left[keep], delta[keep]
    right = None if right is None else right[keep]
    q_left = q_left[keep]
    q_right = None if q_right is None else q_right[keep]
    delta_right = None if delta_right is None else delta_right[keep]
    counters.bump("survivors", len(tag))
    if not len(tag):
        return

    maxima, squares = batch.matrix.exact_scores(batch, tag, left, right, delta, bound,
                                                screen_rows, options, counters,
                                                delta_right)
    admissible = _admissibility(batch, tag, maxima, squares, eps, batch.l2)
    _, position = best_per_instance(batch, tag, maxima, squares, admissible)

    chosen = position[position < len(tag)]
    instances = torch.nonzero(position < len(tag), as_tuple=True)[0]
    if not len(instances):
        return
    if compound:
        batch.apply_compound(instances, left[chosen], right[chosen],
                             q_left[chosen], q_right[chosen])
    else:
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


class ScreeningRows(NamedTuple):
    """The worst constraints per instance, and what bounds everything below them.

    ``tau`` is the next violation after the kept ones. Every constraint outside
    the kept set is at most ``tau``, which is what lets the sparse backend bound
    the constraints a candidate did not touch without reading them.
    """

    indices: torch.Tensor   # (instances, K)
    values: torch.Tensor    # (instances, K)
    tau: torch.Tensor       # (instances,)


def _per_instance_best(batch, tag, scores, width):
    """Flat indices of each instance's ``width`` lowest-scoring candidates.

    Candidates arrive as one flat list tagged by instance, so selecting per
    instance means laying them out by instance first. One scatter into a padded
    block and one ``topk``, rather than a loop.
    """
    counts = torch.bincount(tag, minlength=batch.n_instances)
    widest = int(counts.max())
    offsets = torch.cumsum(counts, 0) - counts
    within = torch.arange(len(tag), device=batch.device) - offsets[tag]

    padded = torch.full((batch.n_instances, widest), float("inf"),
                        dtype=scores.dtype, device=batch.device)
    padded[tag, within] = scores
    take = min(width, widest)
    chosen_within = padded.topk(take, dim=1, largest=False).indices
    flat = offsets[:, None] + chosen_within
    # Instances with fewer candidates than `take` would index past their block.
    return flat, chosen_within < counts[:, None]


def compound_pass(batch, active, screen_rows, options, counters):
    """One batched compound pass: move two variables at once. Returns who improved.

    This is the move a swap only pretends to be. A swap makes two variables
    exchange levels, which fixes the two steps to be equal and opposite and keeps
    the multiset of assigned levels -- a structure that has nothing to do with
    which constraint is binding. Here the two steps are independent, and the pairs
    are drawn from the single moves that lose the least, so the pass can leave a
    point where every single move is worse.

    Pairs are taken from the ``compound_width`` least damaging single candidates
    rather than from all of them: all pairs would be quadratic in the variable
    count, and a pair built from two badly damaging moves is not going to win.
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

    singles, _ = batch.matrix.exact_scores(batch, tag, left, None, delta,
                                           batch.objective, screen_rows, options,
                                           counters)
    picks, valid = _per_instance_best(batch, tag, singles, options.compound_width)
    width = picks.shape[1]
    if width < 2:
        return improved

    # Every unordered pair of the kept candidates, for every instance at once.
    first, second = torch.triu_indices(width, width, offset=1, device=batch.device)
    rows = torch.arange(batch.n_instances, device=batch.device)[:, None]
    a = picks[rows, first[None, :]]
    b = picks[rows, second[None, :]]
    usable = (valid[rows, first[None, :]] & valid[rows, second[None, :]]
              & active[:, None])
    a, b, usable = a.reshape(-1), b.reshape(-1), usable.reshape(-1)
    a, b = a[usable], b[usable]
    if not len(a):
        return improved

    # A pair has to move two different variables to be a compound move at all.
    distinct = left[a] != left[b]
    a, b = a[distinct], b[distinct]
    if not len(a):
        return improved
    counters.bump("compound_candidates", len(a))

    _evaluate_and_apply(batch, tag[a], left[a], left[b], delta[a],
                        level[a], level[b], screen_rows, options, counters, improved,
                        delta_right=delta[b], compound=True)
    return improved


def screening_rows(batch, n_filters) -> ScreeningRows:
    """Return the ``n_filters`` most violated constraints per instance.

    Keyed on the violation, not on the raw value of ``A x``: with one-sided or
    asymmetric bounds the largest values are not the most violated constraints,
    and screening against the wrong set silently discards good candidates.
    """
    v = batch.violation_of(batch.y).T                       # (instances, constraints)
    k = min(n_filters, batch.n_constraints)
    top = v.topk(min(k + 1, batch.n_constraints), dim=1)
    if top.indices.shape[1] > k:
        tau = top.values[:, k]
    else:
        # Nothing lies outside the kept set, so nothing needs bounding.
        tau = torch.full((batch.n_instances,), float("-inf"), dtype=v.dtype,
                         device=v.device)
    return ScreeningRows(top.indices[:, :k], top.values[:, :k], tau)


def local_search(batch, active, options, counters, deadline=None, max_passes=1000):
    """Descend until no active instance improves.

    A pass is the smallest unit that leaves the batch consistent, so the deadline
    is checked between passes. Without it a single descent can run far past the
    budget, which makes an equal-time comparison meaningless.

    Single-variable moves run before swaps. Swaps preserve the multiset of
    assigned levels, so from a point with the wrong level histogram no sequence
    of swaps can reach a better one.
    """
    if not (options.single_variable_moves or options.swap_moves
            or options.compound_moves):
        raise ValueError("no move class is enabled; set single_variable_moves or "
                         "swap_moves")
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
        if options.swap_moves:
            improved |= swap_pass(batch, working, rows, options, counters)
        if options.compound_moves:
            # Last, and only for instances nothing else could move: a compound
            # pass costs a full single-candidate scoring before it starts.
            stuck = working & ~improved
            if bool(stuck.any()):
                rows = screening_rows(batch, n_filters)
                improved |= compound_pass(batch, stuck, rows, options, counters)
        counters.bump("passes")
        working = working & improved
        if not bool(working.any()):
            break
    return batch


def kick_sizes(batch, destroy_rate, counts=None):
    """How many variables each instance's kick moves, as ``(sizes, widest)``.

    ``counts`` lets the caller widen the kick per instance; without it every
    instance gets the same size, which is the original behaviour.
    """
    base = max(1, int(round(destroy_rate * batch.n_variables)))
    if counts is None:
        sizes = torch.full((batch.n_instances,), base, dtype=torch.long,
                           device=batch.device)
    else:
        sizes = counts.clamp(1, batch.n_variables)
    return sizes, int(sizes.max())


def _chosen_mask(scores, sizes, widest):
    """Top-``sizes[i]`` positions per instance, as indices plus a validity mask.

    One ``topk`` at the widest size and a rank comparison, rather than a loop per
    instance: the kick has to stay a handful of kernels however ragged the sizes
    become.
    """
    chosen = scores.topk(widest, dim=1).indices
    rank = torch.arange(widest, device=scores.device)[None, :]
    return chosen, rank < sizes[:, None]


def perturb(batch, active, destroy_rate, how="random", counts=None):
    """Kick each active instance out of its local optimum.

    ``how`` selects the kick: see ``Options.perturbation``. ``counts`` optionally
    sets a per-instance kick size.
    """
    if how == "active":
        return perturb_active(batch, active, destroy_rate, counts)
    if how != "random":
        raise ValueError(f"unknown perturbation {how!r}; expected 'random' or 'active'")
    return perturb_random(batch, active, destroy_rate, counts)


def perturb_active(batch, active, destroy_rate, counts=None):
    """Move variables of the worst constraint, each the way that relieves it.

    A blind kick is mostly wasted: the objective is the largest violation, so it
    is decided by one constraint at a time, and moving variables that constraint
    does not involve leaves it exactly where it was. This picks the worst
    constraint per instance and pushes a random subset of *its* variables in the
    direction that reduces it -- lowering ``a.x`` where the upper bound is
    exceeded, raising it where the lower bound is missed.

    The step is still one level and the subset is still random, so this changes
    where the kick lands, not how hard it hits.
    """
    device = batch.device
    v = batch.violation_of(batch.y)
    worst = v.argmax(dim=0)                                  # (instances,)
    row = batch.matrix.rows_of(worst)                        # (instances, variables)

    # Which way relieves the constraint: over its upper bound, reduce a.x.
    y_at = batch.y[worst, torch.arange(batch.n_instances, device=device)]
    upper_at = batch.upper[worst, torch.arange(batch.n_instances, device=device)]
    over = (y_at > upper_at)[:, None]
    # Raising x_j raises a.x when a_j > 0. So to reduce a.x, step down where the
    # coefficient is positive and up where it is negative; invert when under.
    direction = torch.where(over, -row.sign(), row.sign()).long()

    # A variable already at the end of its domain in the relieving direction
    # cannot move, and picking one wastes the whole kick. With the default kick
    # size of a single variable that is not a small loss: measured on a
    # near-rank-deficient instance, the kick changed the objective by exactly
    # 0.000 and none of 40 kicks was ever accepted.
    assignable = (~batch.fixed_mask) & active[:, None] & (direction != 0)
    scores = torch.rand(batch.x_idx.shape, generator=batch.generator, device=device)
    scores = torch.where(assignable, scores, torch.full_like(scores, -1.0))
    sizes, widest = kick_sizes(batch, destroy_rate, counts)
    chosen, use = _chosen_mask(scores, sizes, widest)

    current = torch.gather(batch.x_idx, 1, chosen)
    step = torch.gather(direction, 1, chosen)
    # A variable outside the constraint's support has no direction; leave it.
    proposed = (current + step).clamp(0, batch.n_levels - 1)
    proposed = torch.where(use & active[:, None], proposed, current)
    batch.x_idx.scatter_(1, chosen, proposed)
    batch.refresh()


def perturb_random(batch, active, destroy_rate, counts=None):
    """Move a random subset of each active instance's variables by one level."""
    assignable = (~batch.fixed_mask) & active[:, None]
    scores = torch.rand(batch.x_idx.shape, generator=batch.generator, device=batch.device)
    scores = torch.where(assignable, scores, torch.full_like(scores, -1.0))
    sizes, widest = kick_sizes(batch, destroy_rate, counts)
    chosen, use = _chosen_mask(scores, sizes, widest)

    steps = torch.randint(-1, 2, chosen.shape, generator=batch.generator,
                          device=batch.device)
    current = torch.gather(batch.x_idx, 1, chosen)
    proposed = (current + steps).clamp(0, batch.n_levels - 1)
    proposed = torch.where(use & active[:, None], proposed, current)
    batch.x_idx.scatter_(1, chosen, proposed)
    batch.refresh()
