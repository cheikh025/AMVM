"""Correctness coverage for the batched multi-row solver.

The batched path is a different program from the single-row solver, not a
refactor of it, so it is not held to trajectory equality. What it must satisfy is
every hard constraint the single-row path satisfies: residuals that agree with
direct recomputation, values inside the caller's domain, fixed variables left
alone, and a local search that really does leave the batch locally optimal.
"""
import pytest
import torch

from ALNS import batched


def build(device, n_rows=5, n_variables=9, n_samples=17, n_levels=4, seed=0,
          policy="linf_l2_nonincrease"):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_samples, n_variables, generator=generator).to(device)
    weights = torch.randint(0, n_levels, (n_rows, n_variables), generator=generator).to(device)
    steps = torch.linspace(-1, 1, n_levels)
    levels = steps.repeat(n_rows, 1).to(device)
    original = torch.randn(n_rows, n_variables, generator=generator).to(device)
    B_k = (inputs @ original.T).T.contiguous()
    return batched.BatchedRows(inputs, weights, levels, B_k, acceptance_policy=policy)


def test_residual_matches_direct_recomputation(device):
    batch = build(device)
    for row in range(batch.n_rows):
        direct = batch.inputs @ batch.levels[row][batch.weights[row]] - batch.B_k[row]
        assert torch.allclose(batch.residual[:, row], direct, atol=1e-5)
        assert batch.objective[row].item() == pytest.approx(direct.abs().max().item(), abs=1e-5)


def test_candidate_lists_are_the_exact_cross_product(device):
    batch = build(device, seed=3)
    active = torch.ones(batch.n_rows, dtype=torch.bool, device=device)
    q1, q2 = 1, 2
    tag, left, right = batched.candidate_lists(batch, q1, q2, active)

    produced = sorted(zip(tag.tolist(), left.tolist(), right.tolist()))
    expected = sorted(
        (row, i, j)
        for row in range(batch.n_rows)
        for i in range(batch.n_variables) if batch.weights[row, i] == q1
        for j in range(batch.n_variables) if batch.weights[row, j] == q2)
    assert produced == expected


def test_candidate_lists_skip_inactive_rows_and_fixed_variables(device):
    batch = build(device, seed=4)
    batch.fixed_mask[:, 0] = True
    active = torch.ones(batch.n_rows, dtype=torch.bool, device=device)
    active[1] = False

    tag, left, right = batched.candidate_lists(batch, 1, 2, active)
    assert 1 not in tag.tolist()
    assert 0 not in left.tolist() and 0 not in right.tolist()


def test_applied_swap_agrees_with_direct_recomputation(device):
    batch = build(device, seed=7)
    active = torch.ones(batch.n_rows, dtype=torch.bool, device=device)
    screen_rows = batch.residual.abs().T.topk(8, dim=1).indices

    batched.swap_pass(batch, active, screen_rows)

    for row in range(batch.n_rows):
        direct = batch.inputs @ batch.levels[row][batch.weights[row]] - batch.B_k[row]
        assert torch.allclose(batch.residual[:, row], direct, atol=1e-4)


def test_swap_pass_never_worsens_a_row(device):
    batch = build(device, seed=11)
    before = batch.objective.clone()
    active = torch.ones(batch.n_rows, dtype=torch.bool, device=device)
    screen_rows = batch.residual.abs().T.topk(8, dim=1).indices

    batched.swap_pass(batch, active, screen_rows)

    assert torch.all(batch.objective <= before + 1e-5)


def test_local_search_leaves_every_row_locally_optimal(device):
    batch = build(device, seed=13)
    active = torch.ones(batch.n_rows, dtype=torch.bool, device=device)
    batched.local_search(batch, active)

    settled = batch.objective.clone()
    screen_rows = batch.residual.abs().T.topk(8, dim=1).indices
    improved = batched.swap_pass(batch, active, screen_rows)

    assert not bool(improved.any())
    assert torch.allclose(batch.objective, settled, atol=1e-6)


@pytest.mark.parametrize("policy", ["linf", "linf_l2_tiebreak", "linf_l2_nonincrease"])
def test_solve_returns_feasible_weights_and_honest_objectives(device, policy):
    batch = build(device, seed=17, policy=policy)
    weights, objective = batched.solve(
        batch.inputs, batch.weights, batch.levels, batch.B_k, seconds=0.2,
        acceptance_policy=policy)

    assert weights.shape == batch.weights.shape
    assert int(weights.min()) >= 0 and int(weights.max()) < batch.n_levels
    for row in range(batch.n_rows):
        direct = batch.inputs @ batch.levels[row][weights[row]] - batch.B_k[row]
        assert objective[row].item() == pytest.approx(direct.abs().max().item(), abs=1e-4)


def test_solve_never_returns_a_point_worse_than_its_start(device):
    batch = build(device, seed=19)
    start = batch.objective.clone()
    _, objective = batched.solve(batch.inputs, batch.weights, batch.levels, batch.B_k,
                                 seconds=0.2)
    assert torch.all(objective <= start + 1e-5)


def test_solve_leaves_fixed_variables_untouched(device):
    batch = build(device, seed=23)
    fixed = torch.zeros_like(batch.weights, dtype=torch.bool)
    fixed[:, ::3] = True
    original = batch.weights.clone()

    weights, _ = batched.solve(batch.inputs, batch.weights, batch.levels, batch.B_k,
                               seconds=0.2, fixed_mask=fixed)

    assert torch.equal(weights[fixed], original[fixed])


def test_nonuniform_per_row_domains_are_respected(device):
    """Rows may carry different level values, as they do with lookup-table weights."""
    batch = build(device, seed=29)
    levels = torch.stack([torch.tensor([-0.9, -0.1, 0.2, 1.3]),
                          torch.tensor([-1.0, 0.0, 0.5, 0.7]),
                          torch.tensor([-0.5, -0.2, 0.1, 0.4]),
                          torch.tensor([-2.0, -1.0, 1.0, 2.0]),
                          torch.tensor([-0.3, 0.0, 0.3, 0.6])]).to(device)

    weights, objective = batched.solve(batch.inputs, batch.weights, levels, batch.B_k,
                                       seconds=0.2)
    for row in range(batch.n_rows):
        chosen = levels[row][weights[row]]
        assert torch.isin(chosen, levels[row]).all()
        direct = batch.inputs @ chosen - batch.B_k[row]
        assert objective[row].item() == pytest.approx(direct.abs().max().item(), abs=1e-4)
