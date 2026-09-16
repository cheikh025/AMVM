"""Device auto-selection: cuda -> mps -> cpu, unless a device is given explicitly."""
import pytest
import torch

from utils.utils import select_device

_MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def test_select_device_honors_explicit_request():
    # An explicit request always wins, even over available accelerators.
    assert select_device("cpu").type == "cpu"
    assert select_device(torch.device("cpu")).type == "cpu"
    assert select_device("cpu", index=3).type == "cpu"


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"),   # cuda has top priority
    (True, False, "cuda"),
    (False, True, "mps"),   # then Apple Silicon Metal
    (False, False, "cpu"),  # then CPU
])
def test_select_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device().type == expected


def test_select_device_cuda_uses_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = select_device(index=2)
    assert device.type == "cuda"
    assert device.index == 2


@pytest.mark.skipif(not _MPS, reason="Apple Silicon MPS not available")
def test_end_to_end_solve_on_mps():
    from ALNS.ALNS import ALNS
    from alns.stop import MaxIterations

    # CPU inputs and no explicit device: the algorithm must auto-select MPS and
    # move its tensors there on its own.
    solver = ALNS(torch.tensor([0, 1, 0]), torch.tensor([0.0, 1.0, 0.0]), torch.eye(3), 1,
                  discrete_domain=torch.tensor([0.0, 2.0]), B_k=torch.tensor([2.0, 0.0, 1.0]))
    assert solver.torch_device.type == "mps"
    solver.set_stopping_criteria(MaxIterations(5))
    solver.set_LS_operator("S")
    solution = solver.solve()
    assert solution.quantized_weights.device.type == "mps"
    assert torch.isin(solution.quantized_weights.cpu(), torch.tensor([0.0, 2.0])).all()
