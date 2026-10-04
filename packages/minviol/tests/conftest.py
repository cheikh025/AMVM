import os

import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible_randomness():
    torch.manual_seed(9101)


@pytest.fixture
def device():
    name = os.environ.get("MINVIOL_TEST_DEVICE", "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        pytest.fail("MINVIOL_TEST_DEVICE requests CUDA, but CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        pytest.fail("MINVIOL_TEST_DEVICE requests MPS, but MPS is unavailable")
    return torch.device(name)


@pytest.fixture
def feasible_system(device):
    """A random system with a planted solution, so feasibility is known to exist."""
    def build(n_constraints=400, n_variables=32, n_levels=4, slack=0.5, seed=0):
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(n_constraints, n_variables, generator=g).to(device)
        domain = torch.linspace(-1, 1, n_levels).to(device)
        x_star = domain[torch.randint(0, n_levels, (n_variables,), generator=g).to(device)]
        target = A @ x_star
        return A, domain, x_star, target - slack, target + slack
    return build
