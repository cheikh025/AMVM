import pytest
import torch

from ALNS.local_search import LocalSearch


@pytest.mark.parametrize("levels,bits,indices", [
    (None, 1, [0, 1]),
    ([0, 1, 10, 11], 2, [1, 2]),
])
def test_swap_evaluation_matches_direct_objective_and_rolls_back(make_state, levels, bits, indices):
    state = make_state([[1, -2], [3, 0.5]], indices, [0.1, 0.2],
                       levels=levels, bits=bits)
    before_weights = state.weights.clone()
    before_residual = state.signedD_ks.clone()
    search = LocalSearch(state)
    candidate = state.get_quantized_weights().flip(0)
    expected = (state.inputs @ candidate - state.B_k).abs().max().item()
    assert search.evaluate_swap_general([0], [1]) == pytest.approx(expected, abs=1e-5)
    torch.testing.assert_close(state.weights, before_weights)
    torch.testing.assert_close(state.signedD_ks, before_residual, atol=1e-5, rtol=1e-5)
    assert state.changes == []


def test_screened_out_rows_are_safe_for_every_adjacent_swap(make_state):
    # Row 1 has small coefficients and substantial slack, so screening can omit it.
    state = make_state([[1, -2, 3, -1], [0.01, -0.01, 0.02, 0], [2, -3, 1, 1]],
                       [0, 1, 0, 1], [0, 0, 1])
    search = LocalSearch(state)
    search.input_filtering()
    omitted = sorted(set(range(len(state.inputs))) - set(search.useful_input_indices.tolist()))
    assert omitted, "Fixture must exercise row screening"
    for i in (0, 2):
        for j in (1, 3):
            candidate = state.get_quantized_weights().clone()
            candidate[i], candidate[j] = candidate[j].clone(), candidate[i].clone()
            residual = state.inputs @ candidate - state.B_k
            assert torch.all(residual[omitted].abs() < state.objective())


def test_bounded_swap_uses_physical_nonuniform_delta(make_state):
    state = make_state(torch.eye(4), [1, 1, 2, 2], [10, 1, 1, 10],
                       original=[0, 11, 0, 11], levels=[0, 1, 10, 11], bits=2)
    search = LocalSearch(state)
    assert search._perform_swap_bounded(variable_tile=1, row_tile=1,
                                        candidate_chunk=1)
    expected = state.inputs @ state.get_quantized_weights() - state.B_k
    torch.testing.assert_close(state.signedD_ks, expected)
    assert state.objective() == pytest.approx(0)


def test_bounded_swap_accepts_l2_tiebreak_without_linf_gain(make_state):
    # Swapping the two variables ties the infinity norm (2 -> 2) while halving the
    # sum of squares (8 -> 4), so only the L2 tie-break policy should take it.
    matrix = [[2.0, 2.0], [0.0, 2.0]]
    tie = make_state(matrix, [0, 1], [0.0, 0.0], acceptance_policy="linf_l2_tiebreak")
    assert tie.objective() == pytest.approx(2.0)
    assert tie.L2_norm.item() == pytest.approx(8.0)

    search = LocalSearch(tie)
    assert search._perform_swap_bounded(variable_tile=1, row_tile=1, candidate_chunk=1)
    expected = tie.inputs @ tie.get_quantized_weights() - tie.B_k
    torch.testing.assert_close(tie.signedD_ks, expected)
    assert tie.weights.tolist() == [1, 0]
    assert tie.objective() == pytest.approx(2.0)
    assert tie.L2_norm.item() == pytest.approx(4.0)

    # The pure infinity-norm policy must reject a swap that does not lower it.
    pure = make_state(matrix, [0, 1], [0.0, 0.0], acceptance_policy="linf")
    pure_search = LocalSearch(pure)
    assert pure_search._perform_swap_bounded(variable_tile=1, row_tile=1,
                                             candidate_chunk=1) is False
    assert pure.weights.tolist() == [0, 1]
