import numpy as np
import pytest
import torch

from ALNS.local_search import LocalSearch
from ALNS.remove_operators import random_remove
from ALNS.repair_operators import greedy_repair, random_repair


class OrderedRng:
    """Keep the counterexample's repair order deterministic."""
    def shuffle(self, values):
        pass


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B01: greedy repair evaluates subsequent moves on stale residuals",
                   raises=AssertionError)
def test_greedy_repair_preserves_sequential_improvement(make_state):
    state = make_state([[1.8, 1.7]], [0, 0], [1])
    state.removed_array[:] = True
    initial = state.objective()
    result = greedy_repair(state, OrderedRng())
    exact = (result.inputs @ result.get_quantized_weights() - result.B_k).abs().max().item()
    # The first move gives 0.8. The second must be rejected (combined value 2.5).
    assert exact <= initial
    assert exact == pytest.approx(0.8)


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B03: greedy repair changes a preserved outlier's index and residual",
                   raises=AssertionError)
def test_greedy_repair_keeps_outlier_residual_consistent(make_state):
    state = make_state([[1, 0]], [0, 0], [2], original=[1, 0],
                       keep_outliers=True, outlier_range=0.5)
    state.removed_array[:] = True
    result = greedy_repair(state, OrderedRng())
    exact = result.inputs @ result.get_quantized_weights() - result.B_k
    assert torch.allclose(result.signedD_ks, exact)


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B03: 1-OPT moves preserved outliers but decoding restores their values",
                   raises=AssertionError)
def test_one_opt_keeps_outlier_residual_consistent(make_state):
    state = make_state([[1, 0]], [0, 0], [2], original=[1, 0],
                       keep_outliers=True, outlier_range=0.5)
    LocalSearch(state).local_search_change_weights()
    exact = state.inputs @ state.get_quantized_weights() - state.B_k
    assert torch.allclose(state.signedD_ks, exact)


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B06: floor(0.005*n) removes no variables when n < 200",
                   raises=AssertionError)
def test_random_destroy_perturbs_small_nonempty_problem(make_state):
    state = make_state([[1, 2]], [0, 1], [0])
    result = random_remove(state, np.random.RandomState(1))
    assert int(result.removed_array.sum()) >= 1


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B07: random destroy samples with replacement and removes fewer variables",
                   raises=AssertionError)
def test_random_destroy_selects_requested_number_of_distinct_variables(make_state, monkeypatch):
    import ALNS.remove_operators as operators
    monkeypatch.setattr(operators, "DESTROY_RATE", 1.0)
    state = make_state([list(range(20))], [0] * 20, [0])
    result = random_remove(state, np.random.RandomState(1))
    assert int(result.removed_array.sum()) == 20


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B08: local search swallows unrelated RuntimeError exceptions",
                   raises=AssertionError)
@pytest.mark.parametrize("entry", ["run", "local_search_optimized_swap"])
def test_unexpected_runtime_error_propagates(make_state, monkeypatch, entry):
    state = make_state([[1, 2]], [0, 1], [0])
    state.LS_op = "S"
    search = LocalSearch(state)
    def fail():
        raise RuntimeError("injected shape mismatch")
    target = "local_search_optimized_swap" if entry == "run" else "perform_swap"
    monkeypatch.setattr(search, target, fail)
    caught = None
    try:
        getattr(search, entry)()
    except RuntimeError as exc:
        caught = exc
    assert caught is not None, "Unexpected runtime errors must not become successful solves"
    assert "shape mismatch" in str(caught)


@pytest.mark.policy
def test_random_repair_currently_rejects_linf_improvement_with_worse_l2(make_state):
    # Current residual [2,0] -> candidate [1.5,1.5]: better max, worse sum of squares.
    state = make_state([[-0.5, 0], [1.5, 0]], [0, 0], [-2, 0])
    state.removed_array[0] = True
    class IncreaseRng:
        def randint(self, **kwargs):
            return 1
    result = random_repair(state, IncreaseRng())
    exact = (result.inputs @ result.get_quantized_weights() - result.B_k).abs().max().item()
    assert exact == pytest.approx(1.5)
    assert result.objective() == float("inf")
