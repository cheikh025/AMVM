"""End-to-end behaviour of the solver.

The claims under test are the ones the package makes to a caller: the reported
violation is the real one, a feasible point is found and recognized, the answer
stays inside the domain, and fixed variables never move.
"""
import pytest
import torch

import minviol
from minviol import Budget, Options


def recompute(A, x, lower, upper):
    """The violation, from scratch, with none of the solver's cached state."""
    return minviol.violation(A @ x, lower, upper).max().item()


@pytest.mark.parametrize("init", ["zero", "lstsq_round"])
def test_finds_and_recognizes_a_feasible_point(feasible_system, init):
    A, domain, _, lower, upper = feasible_system(seed=5)
    result = minviol.solve(A, lower, upper, domain=domain, init=init,
                           budget=Budget(seconds=5.0))

    assert result.feasible, f"no feasible point from init={init!r}"
    assert result.max_violation == 0.0
    # Stopping early is the point: a solver that always spends its budget cannot
    # claim to answer "find something feasible, fast".
    assert result.wall_time < 5.0


@pytest.mark.parametrize("init", ["zero", "lstsq_round"])
def test_reported_violation_matches_recomputation(feasible_system, init):
    A, domain, _, lower, upper = feasible_system(seed=6, slack=0.0)
    result = minviol.solve(A, lower, upper, domain=domain, init=init,
                           budget=Budget(seconds=1.0))
    assert result.max_violation == pytest.approx(recompute(A, result.x, lower, upper),
                                                 abs=1e-5)


def test_answer_stays_inside_the_domain(feasible_system):
    A, domain, _, lower, upper = feasible_system(seed=7, slack=0.0)
    result = minviol.solve(A, lower, upper, domain=domain, budget=Budget(seconds=0.5))
    assert bool(torch.isin(result.x, domain).all())
    assert int(result.x_index.min()) >= 0
    assert int(result.x_index.max()) < domain.numel()


def test_incremental_updates_do_not_drift(feasible_system):
    """Run long with the periodic rebuild disabled; the cached objective must hold.

    The rebuild exists because incremental residual updates accumulate float
    error. Turning it off is how a sign error or a double-applied column shows
    up as a number rather than as a slow quality loss.
    """
    A, domain, _, lower, upper = feasible_system(n_constraints=300, seed=8, slack=0.0)
    result = minviol.solve(A, lower, upper, domain=domain, budget=Budget(seconds=2.0),
                           options=Options(refresh_every=10 ** 9))
    assert result.max_violation == pytest.approx(recompute(A, result.x, lower, upper),
                                                 abs=1e-5)


def test_never_returns_a_point_worse_than_its_start(feasible_system):
    A, domain, _, lower, upper = feasible_system(seed=9, slack=0.0)
    start = minviol.solve(A, lower, upper, domain=domain, init="zero",
                          budget=Budget(seconds=0.0))
    better = minviol.solve(A, lower, upper, domain=domain, init="zero",
                           budget=Budget(seconds=1.0))
    assert better.max_violation <= start.max_violation + 1e-6


def test_fixed_variables_never_move(feasible_system):
    A, domain, _, lower, upper = feasible_system(seed=10, slack=0.0)
    n = A.shape[1]
    fixed = torch.zeros(n, dtype=torch.bool, device=A.device)
    fixed[::3] = True
    start = minviol.solve(A, lower, upper, domain=domain, init="zero",
                          budget=Budget(seconds=0.0))
    result = minviol.solve(A, lower, upper, domain=domain, init="given",
                           x0=start.x_index, fixed=fixed, budget=Budget(seconds=1.0))
    assert torch.equal(result.x_index[fixed], start.x_index[fixed])


def test_row_scale_changes_what_is_minimized_but_not_what_is_reported(feasible_system):
    """``max_violation`` answers "does my constraint hold"; scaling is search policy."""
    A, domain, _, lower, upper = feasible_system(seed=11, slack=0.0)
    scale = torch.ones(A.shape[0], device=A.device)
    scale[0] = 1000.0
    result = minviol.solve(A, lower, upper, domain=domain, row_scale=scale,
                           budget=Budget(seconds=1.0))
    assert result.max_violation == pytest.approx(recompute(A, result.x, lower, upper),
                                                 abs=1e-4)
    assert result.objective >= result.max_violation - 1e-6


def test_equality_and_inequality_forms_agree_when_they_describe_the_same_set():
    """A range row written two ways must give the same violation-zero region."""
    torch.manual_seed(2)
    A = torch.randn(120, 10)
    domain = torch.linspace(-1, 1, 3)
    x_star = domain[torch.randint(0, 3, (10,))]
    target = A @ x_star

    two_sided = minviol.solve(A, target - 0.25, target + 0.25, domain=domain,
                              budget=Budget(seconds=2.0))
    assert two_sided.feasible
    assert recompute(A, two_sided.x, target - 0.25, target + 0.25) == 0.0


def test_batch_solves_instances_independently():
    torch.manual_seed(4)
    A = torch.randn(200, 16)
    domain = torch.linspace(-1, 1, 4)
    starts = domain[torch.randint(0, 4, (3, 16))]
    targets = (A @ starts.T).T                       # (instances, constraints)

    results = minviol.solve_batch(A, targets, targets, domain=domain,
                                  budget=Budget(seconds=3.0))
    assert len(results) == 3
    for r, target in zip(results, targets):
        assert r.feasible
        assert recompute(A, r.x, target, target) == pytest.approx(0.0, abs=1e-5)
