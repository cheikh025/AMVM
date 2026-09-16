import os
import random

import numpy as np
import pytest
import torch

from ALNS.State import State


@pytest.fixture(autouse=True)
def reproducible_randomness():
    random.seed(9101)
    np.random.seed(9101)
    torch.manual_seed(9101)


@pytest.fixture
def device():
    name = os.environ.get("AMVM_TEST_DEVICE", "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        pytest.fail("AMVM_TEST_DEVICE requests CUDA, but CUDA is unavailable")
    return torch.device(name)


@pytest.fixture
def make_state(device):
    def factory(A, indices, b, *, original=None, bits=1, levels=None,
                keep_outliers=False, outlier_range=0.0, dtype=torch.float32,
                acceptance_policy="linf"):
        A = torch.as_tensor(A, dtype=dtype, device=device)
        if original is None:
            original = torch.linspace(0, 1, A.shape[1], dtype=dtype, device=device)
        return State(
            inputs=A,
            weights=torch.as_tensor(indices, dtype=torch.long, device=device),
            original_weights=torch.as_tensor(original, dtype=dtype, device=device),
            B_k=torch.as_tensor(b, dtype=dtype, device=device),
            nQuantization=bits,
            num_partial=A.shape[0],
            LS_op=None,
            torch_device=device,
            use_squeezellm=levels is not None,
            squeezellm_LUT=(None if levels is None else
                           torch.as_tensor(levels, dtype=dtype, device=device)),
            keep_outliers=keep_outliers,
            outlier_range=outlier_range,
            acceptance_policy=acceptance_policy,
        )
    return factory
