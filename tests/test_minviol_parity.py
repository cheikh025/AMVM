"""minviol must reproduce this repository's batched engine, move for move.

minviol generalizes the objective from ``max|Ax - b|`` to the violation of
``lower <= Ax <= upper``. Setting ``lower == upper == b`` is supposed to recover
the original exactly, so this is the test that says the generalization changed
nothing. It compares trajectories rather than final objectives: two searches can
reach the same number by different routes, and a difference in the route is a
difference in the algorithm.

The comparison is driven iteration by iteration rather than through ``solve``,
because ``solve`` stops on wall-clock time and the number of iterations that fits
in a second is not reproducible.
"""
import pytest
import torch

from ALNS import batched

minviol = pytest.importorskip("minviol")
from minviol import engine as mv_engine            # noqa: E402
from minviol.backends import DenseMatrix           # noqa: E402
from minviol.counters import Counters              # noqa: E402
from minviol.options import Options                # noqa: E402
from minviol.problem import Batch                  # noqa: E402

POLICIES = ["linf", "linf_l2_tiebreak", "linf_l2_nonincrease"]


def instance(device, n_rows=5, n_variables=9, n_samples=17, n_levels=4, seed=0):
    """A checked-in instance: no captured layer weights, reproducible from a seed."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n_samples, n_variables, generator=generator).to(device)
    weights = torch.randint(0, n_levels, (n_rows, n_variables),
                            generator=generator).to(device)
    levels = torch.linspace(-1, 1, n_levels).repeat(n_rows, 1).to(device)
    original = torch.randn(n_rows, n_variables, generator=generator).to(device)
    B_k = (inputs @ original.T).T.contiguous()
    return inputs, weights, levels, B_k


def paired(device, policy, seed=0, fixed_mask=None, **kwargs):
    """Build the same instance on both engines, configured to agree."""
    inputs, weights, levels, B_k = instance(device, seed=seed, **kwargs)
    old = batched.BatchedRows(inputs, weights, levels, B_k, acceptance_policy=policy,
                              fixed_mask=fixed_mask, seed=9101)
    bounds = B_k.T.contiguous()
    new = Batch(DenseMatrix(inputs), weights, levels, bounds, bounds,
                fixed_mask=fixed_mask, acceptance=policy, seed=9101)
    options = Options(
        acceptance=policy,
        # The original threshold is absolute; minviol defaults to relative, which
        # would change acceptance decisions and so diverge the trajectory.
        improvement_tol=batched.EPSILON, relative_improvement=False,
        # The original engine has swaps only, a uniformly random kick of fixed
        # width, and no escalation. minviol's own defaults are none of those --
        # they are tuned for general constraint systems -- so every one is pinned
        # here rather than inherited.
        single_variable_moves=False, swap_moves=True,
        perturbation="random", kick_escalation=False,
        n_filters=100, candidate_tile=4096, destroy_rate=0.005,
        prune_first_stage=512, prune_stage_growth=4, prune_min_constraints=0,
    )
    return old, new, options


def assert_same_point(old, new, where):
    assert torch.equal(old.weights, new.x_idx), f"points diverged {where}"
    assert torch.allclose(old.objective, new.objective, atol=1e-6), \
        f"objectives diverged {where}"


@pytest.mark.parametrize("policy", POLICIES)
def test_initial_state_matches(device, policy):
    old, new, _ = paired(device, policy)
    assert torch.allclose(old.residual, new.y - new.lower, atol=1e-6)
    assert_same_point(old, new, "at the start")


@pytest.mark.parametrize("policy", POLICIES)
def test_one_descent_takes_the_same_moves(device, policy):
    old, new, options = paired(device, policy, seed=7)
    counters = Counters()
    active_old = torch.ones(old.n_rows, dtype=torch.bool, device=device)
    active_new = torch.ones(new.n_instances, dtype=torch.bool, device=device)

    batched.local_search(old, active_old, n_filters=100, candidate_tile=4096)
    mv_engine.local_search(new, active_new, options, counters)

    assert_same_point(old, new, "after one descent")


@pytest.mark.parametrize("policy", POLICIES)
def test_full_trajectory_matches_over_many_iterations(device, policy):
    """Perturb and descend in lockstep; any divergence compounds, so check every step."""
    old, new, options = paired(device, policy, seed=11)
    counters = Counters()
    active_old = torch.ones(old.n_rows, dtype=torch.bool, device=device)
    active_new = torch.ones(new.n_instances, dtype=torch.bool, device=device)

    batched.local_search(old, active_old, n_filters=100, candidate_tile=4096)
    mv_engine.local_search(new, active_new, options, counters)

    for step in range(25):
        batched.destroy_and_repair(old, active_old, destroy_rate=0.005)
        mv_engine.perturb(new, active_new, destroy_rate=0.005)
        assert_same_point(old, new, f"after perturbation {step}")

        batched.local_search(old, active_old, n_filters=100, candidate_tile=4096)
        mv_engine.local_search(new, active_new, options, counters)
        assert_same_point(old, new, f"after descent {step}")


def test_fixed_variables_match(device):
    fixed = torch.zeros(5, 9, dtype=torch.bool, device=device)
    fixed[:, ::3] = True
    old, new, options = paired(device, "linf", seed=13, fixed_mask=fixed)
    counters = Counters()
    active_old = torch.ones(old.n_rows, dtype=torch.bool, device=device)
    active_new = torch.ones(new.n_instances, dtype=torch.bool, device=device)

    for _ in range(5):
        batched.local_search(old, active_old, n_filters=100, candidate_tile=4096)
        mv_engine.local_search(new, active_new, options, counters)
        assert_same_point(old, new, "with fixed variables")
        batched.destroy_and_repair(old, active_old, destroy_rate=0.005)
        mv_engine.perturb(new, active_new, destroy_rate=0.005)


def test_nonuniform_per_instance_domains_match(device):
    """Instances may carry different level values, as lookup-table weights do."""
    inputs, weights, _, B_k = instance(device, seed=29)
    levels = torch.stack([torch.tensor([-0.9, -0.1, 0.2, 1.3]),
                          torch.tensor([-1.0, 0.0, 0.5, 0.7]),
                          torch.tensor([-0.5, -0.2, 0.1, 0.4]),
                          torch.tensor([-2.0, -1.0, 1.0, 2.0]),
                          torch.tensor([-0.3, 0.0, 0.3, 0.6])]).to(device)
    old = batched.BatchedRows(inputs, weights, levels, B_k, acceptance_policy="linf",
                              seed=9101)
    bounds = B_k.T.contiguous()
    new = Batch(DenseMatrix(inputs), weights, levels, bounds, bounds,
                acceptance="linf", seed=9101)
    options = Options(acceptance="linf", improvement_tol=batched.EPSILON,
                      relative_improvement=False, single_variable_moves=False,
                      swap_moves=True, perturbation="random", kick_escalation=False,
                      n_filters=100, candidate_tile=4096, prune_min_constraints=0)
    counters = Counters()
    active_old = torch.ones(old.n_rows, dtype=torch.bool, device=device)
    active_new = torch.ones(new.n_instances, dtype=torch.bool, device=device)

    for _ in range(10):
        batched.local_search(old, active_old, n_filters=100, candidate_tile=4096)
        mv_engine.local_search(new, active_new, options, counters)
        assert_same_point(old, new, "with per-instance domains")
        batched.destroy_and_repair(old, active_old, destroy_rate=0.005)
        mv_engine.perturb(new, active_new, destroy_rate=0.005)


class TestAdapter:
    """The adapter that lets the quantization path run on minviol."""

    @staticmethod
    def args(device, seed=31):
        inputs, weights, levels, B_k = instance(device, seed=seed)
        return inputs, weights, levels, B_k

    @pytest.mark.parametrize("policy", POLICIES)
    def test_adapter_returns_what_the_original_returns(self, device, policy):
        from ALNS import minviol_engine

        inputs, weights, levels, B_k = self.args(device)
        solution, objective = minviol_engine.solve(
            inputs, weights, levels, B_k, seconds=0.2, acceptance_policy=policy)

        assert solution.shape == weights.shape
        assert int(solution.min()) >= 0 and int(solution.max()) < levels.shape[1]
        for row in range(weights.shape[0]):
            direct = inputs @ levels[row][solution[row]] - B_k[row]
            assert objective[row].item() == pytest.approx(direct.abs().max().item(),
                                                          abs=1e-4)

    def test_adapter_never_returns_a_point_worse_than_its_start(self, device):
        from ALNS import minviol_engine

        inputs, weights, levels, B_k = self.args(device, seed=37)
        start = batched.BatchedRows(inputs, weights, levels, B_k).objective.clone()
        _, objective = minviol_engine.solve(inputs, weights, levels, B_k, seconds=0.2)
        assert torch.all(objective <= start + 1e-5)

    def test_adapter_leaves_fixed_variables_untouched(self, device):
        from ALNS import minviol_engine

        inputs, weights, levels, B_k = self.args(device, seed=41)
        fixed = torch.zeros_like(weights, dtype=torch.bool)
        fixed[:, ::3] = True
        solution, _ = minviol_engine.solve(inputs, weights, levels, B_k, seconds=0.2,
                                           fixed_mask=fixed)
        assert torch.equal(solution[fixed], weights[fixed])

    def test_the_dispatcher_reaches_the_minviol_engine_when_the_flag_is_set(self,
                                                                           monkeypatch):
        """The flag has to change which engine runs, not just which one is imported."""
        import full_layer
        from ALNS import minviol_engine, tuning as alns_tuning

        seen = {}

        def record(inputs, weights, levels, B_k, seconds, **kwargs):
            seen["called"] = True
            return torch.zeros_like(weights), torch.zeros(weights.shape[0])

        monkeypatch.setattr(alns_tuning, "MINVIOL_ENGINE", True)
        monkeypatch.setattr(minviol_engine, "solve", record)
        inputs, weights, levels, B_k = self.args(torch.device("cpu"))

        config = type("Config", (), dict(
            use_gptq=False, use_squeezellm=False, keep_outliers=False, nQuantized=2,
            seconds=0.01, save_weights=False, acceptance_policy="linf", seed=1))()
        full_layer.quantize_indices_batched([0, 1], inputs, weights.float(), config)

        assert seen.get("called"), "the flag did not route to the minviol engine"
