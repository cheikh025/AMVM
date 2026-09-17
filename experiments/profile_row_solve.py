"""Profile one ALNS row solve: phase timings, operation counts, synchronizations.

Answers "where does an iteration go" rather than "how fast is it". The wall-clock
numbers printed here carry profiler overhead, which scales with operation count,
so never quote them as speed results: use ``bench_row_solve.py --mode speed`` for
that. What this script is for is the operation and synchronization counts, and
the per-phase split.

    experiments/profile_row_solve.py --instance experiments/data/<layer>.pt --iterations 10
"""

import argparse
import collections
import functools
import pathlib
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "GPU"))

from alns.stop import MaxIterations  # noqa: E402

from ALNS import counters, local_search, remove_operators, repair_operators, tuning  # noqa: E402
from ALNS.ALNS import ALNS  # noqa: E402
from ALNS.State import State  # noqa: E402
from RoundToNearest import FindNearest  # noqa: E402
from bench_row_solve import load_instance, synchronize  # noqa: E402
from utils.utils import select_device  # noqa: E402


def instrument(device):
    """Wrap the solver phases with synchronizing timers. Returns (calls, seconds)."""
    calls, seconds = collections.Counter(), collections.Counter()

    def wrap(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            synchronize(device)
            started = time.perf_counter()
            result = original(*args, **kwargs)
            synchronize(device)
            seconds[label] += time.perf_counter() - started
            calls[label] += 1
            return result

        setattr(owner, name, timed)

    wrap(local_search.LocalSearch, "run", "local_search")
    wrap(local_search.LocalSearch, "perform_swap", "swap_pass")
    wrap(local_search.LocalSearch, "convert2_sol_dict", "build_level_lists")
    wrap(State, "apply_move", "apply_move")
    wrap(remove_operators, "create_copied_state", "copy_state")
    import ALNS.ALNS as solver_module
    for name in ("random_repair", "greedy_repair"):
        wrap(repair_operators, name, name)
        setattr(solver_module, name, getattr(repair_operators, name))
    for name in ("random_remove", "worst_remove"):
        wrap(remove_operators, name, name)
        setattr(solver_module, name, getattr(remove_operators, name))
    return calls, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="synthetic:M=768:R=32")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--acceptance-policy", default="linf_l2_nonincrease")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = select_device(args.device)
    name, X, W = load_instance(args.instance, args.n, device)
    calls, seconds = instrument(device)
    counters.reset()

    original_row = W[args.row, :].contiguous()
    _, initial = FindNearest(original_row, args.bits, device)
    solver = ALNS(initial, original_row, X, args.bits, seed=args.seed,
                  acceptance_policy=args.acceptance_policy)
    solver.set_torch_device(device)
    solver.set_LS_operator("S")
    solver.set_stopping_criteria(MaxIterations(args.iterations))

    synchronize(device)
    started = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        solution = solver.solve()
    synchronize(device)
    wall = time.perf_counter() - started

    averages = profiler.key_averages()
    operations = sum(entry.count for entry in averages if entry.key.startswith("aten::"))

    def count(key):
        return sum(entry.count for entry in averages if entry.key == key)

    scalar_reads = count("aten::_local_scalar_dense") + count("aten::is_nonzero")
    shape_reads = count("aten::nonzero") + count("aten::_unique2")
    iterations = max(args.iterations, 1)

    print(f"\n=== {name}  device={device}  M={X.shape[1]}  N={X.shape[0]}  "
          f"bits={args.bits}  iterations={args.iterations}")
    print(f"flags: {tuning.describe(device)}")
    print(f"objective {solution.initial_obj:.6f} -> {solution.inf_norm_value:.6f}")
    print(f"wall {wall:.2f}s under the profiler (NOT a speed measurement)")
    print(f"aten operations {operations} total, {operations / iterations:.0f} per iteration")
    print(f"synchronizations {scalar_reads + shape_reads} total, "
          f"{(scalar_reads + shape_reads) / iterations:.0f} per iteration "
          f"({scalar_reads} scalar reads, {shape_reads} shape reads)")
    print(f"{'phase':28s} {'calls':>7s} {'seconds':>9s} {'ms/call':>9s}")
    for label in sorted(seconds, key=lambda key: -seconds[key]):
        print(f"{label:28s} {calls[label]:7d} {seconds[label]:9.3f} "
              f"{seconds[label] / max(calls[label], 1) * 1000:9.2f}")
    print(f"counters: {counters.snapshot()}")
    print("top operations by self CPU time:")
    for entry in sorted(averages, key=lambda item: -item.self_cpu_time_total)[:10]:
        print(f"  {entry.key:40s} n={entry.count:8d} "
              f"self={entry.self_cpu_time_total / 1e3:9.1f} ms")


if __name__ == "__main__":
    main()
