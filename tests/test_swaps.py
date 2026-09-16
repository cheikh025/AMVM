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
