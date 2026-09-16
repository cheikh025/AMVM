import pytest
import torch

from ALNS.State import FULL, PARTIAL_D_SINGLE_CHANGE
from ALNS.remove_operators import create_copied_state


@pytest.mark.parametrize("levels,bits,indices", [
    (None, 1, [0, 1, 0]),
    ([0, 1, 10, 11], 2, [1, 2, 0]),
])
def test_initial_cache_matches_direct_residual(make_state, levels, bits, indices):
    state = make_state([[1, -2, 3], [-2, 1, 0.5]], indices, [0.2, -0.4],
                       levels=levels, bits=bits)
    residual = state.inputs @ state.get_quantized_weights() - state.B_k
    torch.testing.assert_close(state.signedD_ks, residual)
    assert state.objective() == pytest.approx(residual.abs().max().item())
    torch.testing.assert_close(state.L2_norm, residual.square().sum())


@pytest.mark.parametrize("levels,bits,indices", [
    (None, 1, [0, 1]),
    ([0, 1, 10, 11], 2, [1, 2]),
])
def test_committed_move_and_inverse_match_direct_residual(make_state, levels, bits, indices):
    state = make_state([[2, -1], [0.5, 3]], indices, [1.1, -0.5],
                       levels=levels, bits=bits)
    before = state.get_quantized_weights().clone()
    for delta in (1, -1):
        state.weights[0] += delta
        state.changes.append((0, delta))
        state.eval_flag, state.recalculate_flag = FULL, True
        value = state.objective()
        expected = state.inputs @ state.get_quantized_weights() - state.B_k
        torch.testing.assert_close(state.signedD_ks, expected)
        assert value == pytest.approx(expected.abs().max().item(), abs=1e-5)
    torch.testing.assert_close(state.get_quantized_weights(), before)


def test_uniform_move_evaluation_does_not_mutate_state(make_state):
    state = make_state([[2, -1], [1, 3]], [0, 1], [1, 2])
    before_weights, before_residual = state.weights.clone(), state.signedD_ks.clone()
    state.change, state.eval_flag = (1, 0), PARTIAL_D_SINGLE_CHANGE
    value = state.objective()
    expected = before_residual + state.inputs[:, 0]
    assert value == pytest.approx(expected.abs().max().item())
    torch.testing.assert_close(state.weights, before_weights)
    torch.testing.assert_close(state.signedD_ks, before_residual)


def test_nonuniform_move_evaluation_uses_actual_level_difference(make_state):
    state = make_state([[1, 0]], [1, 0], [1], original=[0, 11],
                       levels=[0, 1, 10, 11], bits=2)
    state.change, state.eval_flag = (1, 0), PARTIAL_D_SINGLE_CHANGE
    candidate = state.get_quantized_weights().clone()
    candidate[0] = state.quantization_levels[2]
    expected = (state.inputs @ candidate - state.B_k).abs().max().item()
    assert state.objective() == pytest.approx(expected)


def test_candidate_change_queue_is_independent(make_state):
    state = make_state([[1, 2]], [0, 1], [0])
    candidate = create_copied_state(state, torch.ones_like(state.weights, dtype=torch.bool))
    candidate.changes.append((0, 1))
    python_rng_state = state.python_rng.getstate()
    torch_rng_state = state.torch_generator.get_state().clone()
    candidate.python_rng.random()
    torch.rand(1, generator=candidate.torch_generator, device=state.torch_device)
    assert state.changes == []
    assert state.python_rng.getstate() == python_rng_state
    torch.testing.assert_close(state.torch_generator.get_state(), torch_rng_state)


def test_candidate_weight_and_residual_tensors_are_independent(make_state):
    state = make_state([[1, 2]], [0, 1], [0])
    candidate = create_copied_state(state, torch.ones_like(state.weights, dtype=torch.bool))
    candidate.weights[0] = 1
    candidate.signedD_ks.add_(10)
    assert state.weights[0].item() == 0
    assert state.signedD_ks.item() == 2


def test_fixed_variables_cannot_be_changed(make_state):
    state = make_state([[1, 0]], [0, 0], [2], original=[1, 0],
                       keep_outliers=True, outlier_range=0.5)
    with pytest.raises(ValueError, match="fixed variable"):
        state.apply_move([0], [1])
    assert state.weights.tolist() == [0, 0]


def test_l2_constraint_is_explicit(make_state):
    pure = make_state([[-0.5, 0], [1.5, 0]], [0, 0], [-2, 0],
                      acceptance_policy="linf")
    constrained = make_state([[-0.5, 0], [1.5, 0]], [0, 0], [-2, 0],
                             acceptance_policy="linf_l2_nonincrease")
    residual = pure.move_residual([0], [1])
    linf, l2 = residual.abs().max().item(), residual.square().sum()
    assert pure.accepts(linf, l2)
    assert not constrained.accepts(linf, l2)


def test_incremental_update_preserves_float64_accuracy(make_state):
    state = make_state([[1 / 3, 0]], [0, 0], [0], dtype=torch.float64)
    state.weights[0] = 1
    state.changes.append((0, 1))
    state.eval_flag, state.recalculate_flag = FULL, True
    state.objective()
    exact = state.inputs @ state.get_quantized_weights() - state.B_k
    assert torch.allclose(state.signedD_ks, exact, rtol=1e-12, atol=1e-12)
