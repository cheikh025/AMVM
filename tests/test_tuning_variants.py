"""Regression coverage for the optional solver variants in ``ALNS.tuning``.

Each variant must be proved before it is measured. Host-indexing changes must
return exactly what the per-element code returned. The settled-state skip must
only ever skip a pass that would have found nothing, which is the property that
makes skipping it safe.
"""
import copy

import numpy as np
import pytest
import torch

from ALNS import tuning
from ALNS.local_search import LocalSearch, run_local_search
from ALNS.remove_operators import create_copied_state
from ALNS.repair_operators import current_levels_of, repairable_indices


@pytest.fixture
def settled_state(make_state):
    """A state whose swap neighborhood a local-search pass has already examined."""
    generator = torch.Generator().manual_seed(9101)
    A = torch.randn(48, 24, generator=generator)
    original = torch.randn(24, generator=generator)
    state = make_state(A, torch.randint(0, 2, (24,), generator=generator), torch.zeros(48),
                       original=original, bits=1)
    state.LS_op = "S"
    LocalSearch(state).run()
    return state


def test_repairable_indices_matches_per_element_form(make_state):
    A = torch.randn(6, 8, generator=torch.Generator().manual_seed(7))
    state = make_state(A, torch.zeros(8, dtype=torch.long), torch.zeros(6))
    state.removed_array = torch.tensor([True, False, True, True, False, True, False, True],
                                       device=state.torch_device)
    state.fixed_mask = torch.tensor([False, False, True, False, False, False, False, True],
                                    device=state.torch_device)

    expected = [i for i in range(len(state.removed_array))
                if state.removed_array[i] and not state.fixed_mask[i]]
    assert repairable_indices(state) == expected == [0, 3, 5]


def test_current_levels_matches_per_element_reads(make_state):
    A = torch.randn(4, 5, generator=torch.Generator().manual_seed(11))
    state = make_state(A, torch.tensor([0, 1, 1, 0, 1]), torch.zeros(4))
    indices = [4, 1, 3]
    assert current_levels_of(state, indices) == [int(state.weights[i]) for i in indices]
    assert current_levels_of(state, []) == []


def test_local_search_on_a_settled_state_finds_nothing(settled_state):
    """The claim the skip rests on: a repeat pass on the same point is a no-op."""
    assert settled_state.settled
    before_weights = settled_state.weights.clone()
    before_objective = settled_state.objective_value
    before_moves = settled_state.move_count

    LocalSearch(settled_state).run()

    assert settled_state.move_count == before_moves
    assert torch.equal(settled_state.weights, before_weights)
    assert settled_state.objective_value == before_objective


def test_skip_returns_the_same_point_as_running_the_pass(settled_state, monkeypatch):
    ran = copy.deepcopy(settled_state)
    monkeypatch.setattr(tuning, "SKIP_SETTLED_LOCAL_SEARCH", False)
    run_local_search(ran)

    skipped = copy.deepcopy(settled_state)
    monkeypatch.setattr(tuning, "SKIP_SETTLED_LOCAL_SEARCH", True)
    run_local_search(skipped)

    assert torch.equal(skipped.weights, ran.weights)
    assert skipped.objective_value == pytest.approx(ran.objective_value)


def test_applied_move_unsettles_the_state(settled_state):
    mutable = int(torch.nonzero(~settled_state.fixed_mask, as_tuple=True)[0][0])
    other = 1 - int(settled_state.weights[mutable])
    settled_state.apply_move([mutable], [other])
    assert not settled_state.settled


def test_settled_flag_survives_the_copied_state(settled_state):
    removed = torch.zeros_like(settled_state.removed_array)
    removed[0] = True
    assert create_copied_state(settled_state, removed).settled

    settled_state.settled = False
    assert not create_copied_state(settled_state, removed).settled


def test_a_repair_that_moves_nothing_leaves_the_state_settled(settled_state, monkeypatch):
    """The case the skip is aimed at: the repair applies no move, so the pass is wasted."""
    from ALNS import repair_operators

    removed = torch.zeros_like(settled_state.removed_array)
    removed[0] = True
    candidate = create_copied_state(settled_state, removed)
    monkeypatch.setattr(tuning, "SKIP_SETTLED_LOCAL_SEARCH", True)

    moves_before = candidate.move_count
    repair_operators.greedy_repair(candidate, np.random.RandomState(0))

    if candidate.move_count == moves_before:
        assert candidate.settled


def _swap_instance(make_state, seed, rows=40, columns=32, bits=2):
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(rows, columns, generator=generator)
    original = torch.randn(columns, generator=generator)
    indices = torch.randint(0, 2 ** bits, (columns,), generator=generator)
    b = A @ original
    state = make_state(A, indices, b, original=original, bits=bits,
                       acceptance_policy="linf_l2_nonincrease")
    state.LS_op = "S"
    return state


@pytest.mark.parametrize("seed", [0, 3, 17, 2026])
@pytest.mark.parametrize("variable_tile,row_tile,candidate_chunk", [
    (4, 3, 2),      # ragged tiles that divide no dimension
    (256, 256, 512),  # the recorded baseline configuration
])
def test_device_resident_swap_selects_the_same_move(make_state, monkeypatch, seed,
                                                    variable_tile, row_tile, candidate_chunk):
    """The device-resident pass must choose exactly what the bounded pass chooses."""
    monkeypatch.setattr(tuning, "VARIABLE_TILE", variable_tile)
    monkeypatch.setattr(tuning, "ROW_TILE", row_tile)
    monkeypatch.setattr(tuning, "CANDIDATE_CHUNK", candidate_chunk)
    monkeypatch.setattr(tuning, "BUDGET_ROW_TILE", False)

    bounded_state = _swap_instance(make_state, seed)
    bounded = LocalSearch(bounded_state)
    bounded.delta_q = 1
    bounded_moved = bounded._perform_swap_bounded()

    resident_state = _swap_instance(make_state, seed)
    resident = LocalSearch(resident_state)
    resident.delta_q = 1
    resident_moved = resident._perform_swap_device_resident()

    assert bounded_moved == resident_moved
    assert torch.equal(bounded_state.weights, resident_state.weights)
    assert resident_state.objective_value == pytest.approx(bounded_state.objective_value)
    assert resident_state.L2_norm.item() == pytest.approx(bounded_state.L2_norm.item())


@pytest.mark.parametrize("policy", ["linf", "linf_l2_tiebreak", "linf_l2_nonincrease"])
def test_device_resident_swap_honours_every_policy(make_state, monkeypatch, policy):
    monkeypatch.setattr(tuning, "BUDGET_ROW_TILE", False)
    for seed in (5, 11):
        bounded_state = _swap_instance(make_state, seed)
        bounded_state.acceptance_policy = policy
        resident_state = _swap_instance(make_state, seed)
        resident_state.acceptance_policy = policy

        bounded = LocalSearch(bounded_state)
        bounded.delta_q = 1
        resident = LocalSearch(resident_state)
        resident.delta_q = 1

        assert bounded._perform_swap_bounded() == resident._perform_swap_device_resident()
        assert torch.equal(bounded_state.weights, resident_state.weights)


def test_row_tile_budget_bounds_the_intermediate(monkeypatch):
    """The derived tile must respect the budget and never exceed the row count."""
    monkeypatch.setattr(tuning, "BUDGET_ROW_TILE", True)
    monkeypatch.setattr(tuning, "SWAP_MEMORY_BUDGET_MB", 256)
    chunk, element = 512, 4

    assert tuning.resolve_row_tile(1024, chunk, element) == 1024  # bounded by the rows
    derived = tuning.resolve_row_tile(10 ** 7, chunk, element)
    assert derived * chunk * element * tuning.LIVE_INTERMEDIATES <= 256 * 1024 * 1024
    assert tuning.resolve_row_tile(10 ** 7, chunk, element, ) > 256  # beats the fixed tile

    monkeypatch.setattr(tuning, "SWAP_MEMORY_BUDGET_MB", 8)
    smaller = tuning.resolve_row_tile(10 ** 7, chunk, element)
    assert smaller < derived


@pytest.mark.parametrize("seed", [0, 3, 17, 2026, 31337])
@pytest.mark.parametrize("policy", ["linf", "linf_l2_tiebreak", "linf_l2_nonincrease"])
def test_incumbent_pruning_picks_the_same_swap(make_state, monkeypatch, seed, policy):
    """Dropping candidates that cannot win must not change which swap is chosen.

    A candidate is dropped only once its running maximum has passed the best
    candidate so far, and a running maximum never decreases, so no dropped
    candidate could have been selected. Summation order for the L2 term changes,
    so that comparison allows float reassociation.
    """
    monkeypatch.setattr(tuning, "BUDGET_ROW_TILE", False)
    monkeypatch.setattr(tuning, "PRUNE_FIRST_STAGE", 8)

    reference_state = _swap_instance(make_state, seed, rows=96, columns=32)
    reference_state.acceptance_policy = policy
    monkeypatch.setattr(tuning, "PRUNE_TO_INCUMBENT", False)
    reference = LocalSearch(reference_state)
    reference.delta_q = 1
    reference_moved = reference._perform_swap_device_resident()

    pruned_state = _swap_instance(make_state, seed, rows=96, columns=32)
    pruned_state.acceptance_policy = policy
    monkeypatch.setattr(tuning, "PRUNE_TO_INCUMBENT", True)
    pruned = LocalSearch(pruned_state)
    pruned.delta_q = 1
    pruned_moved = pruned._perform_swap_device_resident()

    assert reference_moved == pruned_moved
    assert torch.equal(pruned_state.weights, reference_state.weights)
    assert pruned_state.objective_value == pytest.approx(reference_state.objective_value,
                                                         rel=1e-6)


def test_pruning_keeps_exact_scores_for_surviving_candidates(make_state, monkeypatch):
    """A bound no candidate reaches must prune nothing and return exact scores."""
    monkeypatch.setattr(tuning, "PRUNE_FIRST_STAGE", 2)
    state = _swap_instance(make_state, 5, rows=64, columns=16)
    search = LocalSearch(state)
    search.delta_q = 1

    left = torch.tensor([0, 1, 2], device=state.torch_device)
    right = torch.tensor([3, 4, 5], device=state.torch_device)
    delta = state.quantization_levels[1] - state.quantization_levels[0]

    exact_max, exact_squares = search._exact_candidate_scores(left, right, delta, 64)
    generous = torch.tensor(float(exact_max.max().item()) + 1.0,
                            device=state.torch_device, dtype=state.inputs.dtype)
    pruned_max, pruned_squares = search._exact_candidate_scores_pruned(
        left, right, delta, generous)

    assert torch.allclose(pruned_max, exact_max, atol=1e-6)
    assert torch.allclose(pruned_squares, exact_squares, rtol=1e-5)


def test_pruning_leaves_dropped_candidates_above_the_bound(make_state, monkeypatch):
    """A dropped candidate must still read as worse than the bound downstream."""
    monkeypatch.setattr(tuning, "PRUNE_FIRST_STAGE", 2)
    state = _swap_instance(make_state, 9, rows=64, columns=16)
    search = LocalSearch(state)
    search.delta_q = 1

    left = torch.tensor([0, 1, 2], device=state.torch_device)
    right = torch.tensor([3, 4, 5], device=state.torch_device)
    delta = state.quantization_levels[1] - state.quantization_levels[0]

    exact_max, _ = search._exact_candidate_scores(left, right, delta, 64)
    strict = torch.tensor(float(exact_max.min().item()) - 1e-6,
                          device=state.torch_device, dtype=state.inputs.dtype)
    partial_max, _ = search._exact_candidate_scores_pruned(left, right, delta, strict)

    # A partial maximum is a lower bound on the exact one and already past the bound.
    assert torch.all(partial_max <= exact_max + 1e-6)
    assert torch.all(partial_max > strict)


def test_pruning_never_drops_a_candidate_that_ties_the_incumbent(make_state, monkeypatch):
    """The drop test is strict, so a candidate that can still tie must survive."""
    monkeypatch.setattr(tuning, "PRUNE_FIRST_STAGE", 1)
    state = _swap_instance(make_state, 13, rows=32, columns=12)
    search = LocalSearch(state)
    search.delta_q = 1

    left = torch.tensor([0, 1], device=state.torch_device)
    right = torch.tensor([2, 3], device=state.torch_device)
    delta = state.quantization_levels[1] - state.quantization_levels[0]

    exact_max, exact_squares = search._exact_candidate_scores(left, right, delta, 32)
    at_the_best = exact_max.min().clone()  # exactly the best candidate's own maximum
    pruned_max, pruned_squares = search._exact_candidate_scores_pruned(
        left, right, delta, at_the_best)

    winner = int(exact_max.argmin())
    assert float(pruned_max[winner]) == pytest.approx(float(exact_max[winner]), abs=1e-6)
    assert float(pruned_squares[winner]) == pytest.approx(float(exact_squares[winner]),
                                                          rel=1e-5)
