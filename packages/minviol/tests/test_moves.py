"""Why the single-variable pass exists.

A swap exchanges the levels of two variables, so it preserves the multiset of
assigned levels. A search built only from swaps can reorder a point but cannot
change how many variables sit at each level -- and from a cold start that is
exactly what is wrong. These tests pin that down rather than leaving it as a
remark in a docstring.
"""
import pytest
import torch

import minviol
from minviol import Budget, Options
from minviol.backends import DenseMatrix
from minviol.counters import Counters
from minviol.options import Options as Opt
from minviol import engine
from minviol.problem import Batch


def binary_instance(n_constraints=300, n_variables=24, seed=0, density=0.5):
    """A binary-domain system whose solution needs a specific number of ones."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_constraints, n_variables, generator=g)
    domain = torch.tensor([0.0, 1.0])
    x_star = (torch.rand(n_variables, generator=g) < density).to(A.dtype)
    target = A @ x_star
    return A, domain, x_star, target


def test_swap_pass_is_inert_when_every_variable_shares_a_level():
    """The cold start is not merely a bad point; it is a point swaps cannot leave."""
    A, domain, _, target = binary_instance()
    x0 = torch.zeros(1, A.shape[1], dtype=torch.long)
    bounds = target[:, None]
    batch = Batch(DenseMatrix(A), x0, domain[None, :], bounds, bounds)
    active = torch.ones(1, dtype=torch.bool)
    options = Opt(prune_min_constraints=0)

    rows = engine.screening_rows(batch, 100)
    improved = engine.swap_pass(batch, active, rows, options, Counters())

    assert not bool(improved.any())
    assert torch.equal(batch.x_idx, x0), "a swap changed a point it cannot reach past"


def test_single_variable_pass_moves_the_same_point():
    A, domain, _, target = binary_instance()
    x0 = torch.zeros(1, A.shape[1], dtype=torch.long)
    bounds = target[:, None]
    batch = Batch(DenseMatrix(A), x0, domain[None, :], bounds, bounds)
    before = batch.objective.clone()
    options = Opt(prune_min_constraints=0)

    rows = engine.screening_rows(batch, 100)
    improved = engine.move_pass(batch, torch.ones(1, dtype=torch.bool), rows, options,
                                Counters())

    assert bool(improved.any())
    assert float(batch.objective) < float(before)


def test_cold_start_descent_is_inert_without_single_variable_moves():
    """The whole descent, not just one pass, stalls at the starting point.

    Measured at the descent rather than end to end. Over a full solve the random
    perturbation also changes the level histogram, one variable at a time, so a
    swap-only search does eventually crawl away from a cold start and the
    end-to-end gap closes. That would make an end-to-end assertion a statement
    about the perturbation's luck. The descent is where the claim is exact:
    without single-variable moves it cannot take a single step.
    """
    A, domain, x_star, target = binary_instance(n_constraints=600, n_variables=128,
                                                seed=3)
    x0 = torch.zeros(1, A.shape[1], dtype=torch.long)
    bounds = target[:, None]

    def descend(single_variable_moves):
        batch = Batch(DenseMatrix(A), x0, domain[None, :], bounds, bounds)
        before = float(batch.objective)
        engine.local_search(batch, torch.ones(1, dtype=torch.bool),
                            Opt(single_variable_moves=single_variable_moves,
                                prune_min_constraints=0), Counters())
        return before, float(batch.objective), int((batch.x_idx == 1).sum())

    before, swaps_only, swap_ones = descend(False)
    _, with_moves, move_ones = descend(True)

    assert swaps_only == before, "a swap-only descent moved a point swaps cannot reach"
    assert swap_ones == 0
    assert with_moves < before * 0.9, (
        f"descent with moves only reached {with_moves:.4g} from {before:.4g}")
    assert move_ones > 0


def test_moves_respect_fixed_variables():
    A, domain, _, target = binary_instance(seed=5)
    n = A.shape[1]
    x0 = torch.zeros(1, n, dtype=torch.long)
    fixed = torch.zeros(1, n, dtype=torch.bool)
    fixed[0, ::2] = True
    bounds = target[:, None]
    batch = Batch(DenseMatrix(A), x0, domain[None, :], bounds, bounds, fixed_mask=fixed)

    rows = engine.screening_rows(batch, 100)
    for _ in range(5):
        engine.move_pass(batch, torch.ones(1, dtype=torch.bool), rows,
                         Opt(prune_min_constraints=0), Counters())
    assert torch.equal(batch.x_idx[fixed], x0[fixed])


def test_move_pass_reaches_a_level_that_is_not_a_neighbour():
    """A four-level domain whose only satisfying value is three levels away."""
    A = torch.tensor([[1.0]])
    domain = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    x0 = torch.zeros(1, 1, dtype=torch.long)
    bounds = torch.tensor([[3.0]])
    batch = Batch(DenseMatrix(A), x0, domain, bounds, bounds)

    engine.move_pass(batch, torch.ones(1, dtype=torch.bool),
                     engine.screening_rows(batch, 1), Opt(prune_min_constraints=0),
                     Counters())

    assert int(batch.x_idx[0, 0]) == 3
    assert float(batch.objective) == pytest.approx(0.0, abs=1e-6)
