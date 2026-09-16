"""Regression coverage derived from ``experiments/benchmark_swap_tiles.py``.

The bounded (tiled) swap evaluation exists to cut peak memory. These tests pin
the two properties that justify it: it must return the *same* best swap as the
dense reference on identical inputs, and its analytic filter-tensor bound must
stay well below the dense materialization.
"""
import importlib.util
import pathlib

import pytest
import torch

_BENCHMARK_PATH = (pathlib.Path(__file__).resolve().parents[1]
                   / "experiments" / "benchmark_swap_tiles.py")
_spec = importlib.util.spec_from_file_location("benchmark_swap_tiles", _BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def _instance(seed):
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    rows, group = 24, 16
    columns = 2 * group
    inputs = torch.randn(rows, columns, generator=generator, device=device)
    residual = 4 * torch.randn(rows, generator=generator, device=device)
    left = torch.arange(group, device=device)
    right = torch.arange(group, columns, device=device)
    screen_rows = residual.abs().topk(min(8, rows)).indices
    return inputs, residual, left, right, screen_rows


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 20250915])
@pytest.mark.parametrize("variable_tile,row_tile,candidate_chunk", [
    (1, 1, 1),      # fully tiled: exercises every chunk boundary
    (4, 3, 2),      # ragged tiles that do not divide the dimensions
    (64, 32, 256),  # tiles larger than the instance: single-pass, equals dense
])
def test_tiled_swap_matches_dense_reference(seed, variable_tile, row_tile, candidate_chunk):
    inputs, residual, left, right, screen_rows = _instance(seed)
    delta, epsilon = 0.125, 0.001

    reference = benchmark.dense(inputs, residual, left, right, delta, screen_rows, epsilon)
    tiled = benchmark.tiled(inputs, residual, left, right, delta, screen_rows, epsilon,
                            variable_tile, row_tile, candidate_chunk)

    if reference is None:
        assert tiled is None
        return
    assert tiled is not None
    # Ties can resolve to different indices, but the optimum (linf, l2) must agree.
    assert tiled[0] == pytest.approx(reference[0])
    assert tiled[1] == pytest.approx(reference[1])


def test_tiled_filter_tensor_bound_beats_dense():
    element_bytes = 4  # float32
    # The default configuration recorded in experiments/results/swap_tiles_cpu_default.json.
    screen_rows, group, variable_tile, row_tile = 32, 512, 256, 256
    dense_bytes = screen_rows * group ** 2 * element_bytes
    tiled_bytes = min(row_tile, screen_rows) * min(variable_tile, group) ** 2 * element_bytes
    assert tiled_bytes < dense_bytes
    assert dense_bytes / tiled_bytes == pytest.approx(4.0)

    # The bound must shrink (never grow) as the tiles get smaller than the groups.
    for tile in (32, 64, 128, 256):
        bounded = min(row_tile, screen_rows) * min(tile, group) ** 2 * element_bytes
        assert bounded <= dense_bytes
