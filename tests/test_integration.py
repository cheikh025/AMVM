import importlib
import sys
import types

import numpy as np
import pytest
import torch
from alns.stop import MaxIterations

from ALNS.ALNS import ALNS


def test_solver_returns_feasible_solution_with_consistent_objective(device):
    A = torch.tensor([[1., 2.], [-1., 3.]], device=device)
    original = torch.tensor([0., 1.], device=device)
    initial = torch.tensor([0, 0], device=device)
    b = torch.tensor([1.2, 0.8], device=device)
    solver = ALNS(initial, original, A, 1)
    solver.set_B_k(b)
    solver.set_torch_device(device)
    solver.set_stopping_criteria(MaxIterations(5))
    result = solver.solve()
    exact = (A @ result.quantized_weights - b).abs().max().item()
    assert result.inf_norm_value == pytest.approx(exact, abs=1e-5)
    assert exact <= b.abs().max().item() + 1e-5
    assert torch.all((result.quantized_weights == 0) | (result.quantized_weights == 1))


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B04: tomography State rebuilds grey levels from SART extrema",
                   raises=AssertionError)
def test_tomography_preserves_prescribed_grey_levels(monkeypatch):
    # ASTRA is an import-only substitute: run_ALNS itself uses no ASTRA operations.
    # Execute the real application wrapper, real State, and real ALNS with zero iterations.
    monkeypatch.setitem(sys.modules, "astra", types.ModuleType("astra"))
    module = importlib.import_module("tomograph")
    monkeypatch.setattr(module, "MAX_RANGE", 255, raising=False)
    monkeypatch.setattr(module, "ALNS_ITERS", 0, raising=False)
    result, _ = module.run_ALNS(np.eye(2, dtype=np.float32),
                              np.array([0, 255], dtype=np.float32),
                              np.array([10, 240], dtype=np.float32), 1, "cpu")
    assert np.isin(result, [0, 255]).all(), f"Solver used incorrect grey levels: {result}"


@pytest.mark.known_bug
@pytest.mark.xfail(reason="B09: retry dispatch reruns all rows, not just failed rows",
                   raises=AssertionError)
def test_quantization_retries_only_failed_rows(tmp_path, monkeypatch):
    import full_layer
    from utils.utils import save_tensors_to_hdf5

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    (tmp_path / "full_data_output" / "tmp").mkdir(parents=True)
    calls = []
    def worker(indices, inputs, weights, config):
        calls.append(list(indices))
        for idx in indices:
            if len(calls) == 1 and idx == 1:
                continue  # Simulate exactly one missing worker output.
            save_tensors_to_hdf5({str(idx): config.weights[idx]},
                                f"full_data_output/tmp/tmp_matrix{idx}.h5")
    monkeypatch.setattr(full_layer, "quantize_indices_concurrently", worker)
    config = types.SimpleNamespace(
        num_gpu=1, weights=np.array([[0., 1.], [1., 0.], [1., 1.]]),
        inputs=np.eye(2), index_iterations=3, rows_per_gpu=2,
        debug_process=False, save_weights=True, DBname="test",
        stored_weights_path=str(tmp_path / "result.h5"), nQuantized=1,
    )
    result = full_layer.quantize_matrix(config)
    np.testing.assert_allclose(result, config.weights)
    assert calls == [[0, 1, 2], [1]]
