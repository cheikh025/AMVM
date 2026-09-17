"""Audit what a solved row actually looks like, and what more search time buys.

Two questions the code and the experiment log cannot answer. What structure do
the returned solutions have, which decides whether the neighborhood is aimed at
the right thing; and how much of the objective is still moving after the
production budget, which is the ceiling on anything that only makes the search
faster.
"""
import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "GPU"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from alns.stop import MaxRuntime  # noqa: E402

from ALNS.ALNS import ALNS  # noqa: E402
from RoundToNearest import FindNearest  # noqa: E402
from bench_row_solve import load_instance, synchronize  # noqa: E402
from utils.utils import select_device  # noqa: E402


def structure(X, original, levels, initial_index, final_index, residual):
    """Structural facts about one solved row."""
    maximum = residual.abs().max()
    changed = (final_index != initial_index)
    steps = (final_index - initial_index).abs()
    column_norm = X.abs().max(dim=0).values

    def share_within(fraction):
        return int((residual.abs() >= fraction * maximum).sum())

    return {
        "samples": int(X.shape[0]),
        "variables": int(X.shape[1]),
        "objective": float(maximum),
        # How many samples are at or near the maximum: the active set of the minimax.
        "samples_at_maximum": share_within(1.0 - 1e-9),
        "samples_within_1pct": share_within(0.99),
        "samples_within_10pct": share_within(0.90),
        "samples_within_50pct": share_within(0.50),
        # How far the solution travelled from round-to-nearest.
        "variables_changed": int(changed.sum()),
        "variables_changed_fraction": float(changed.float().mean()),
        "max_levels_moved": int(steps.max()),
        "variables_moved_more_than_one_level": int((steps > 1).sum()),
        # Which variables can still move at all.
        "variables_at_domain_floor": int((final_index == 0).sum()),
        "variables_at_domain_ceiling": int((final_index == len(levels) - 1).sum()),
        # Are the variables the search moved the ones with the largest columns?
        "mean_column_scale_moved": float(column_norm[changed].mean()) if int(changed.sum()) else 0.0,
        "mean_column_scale_fixed": float(column_norm[~changed].mean()),
        "residual_l2_share_of_top_100": float(
            residual.abs().topk(min(100, len(residual))).values.square().sum()
            / residual.square().sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--acceptance-policy", default="linf_l2_nonincrease")
    parser.add_argument("--out")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = select_device()
    name, X, W = load_instance(args.instance, args.n, device)
    original = W[args.row, :].contiguous()
    _, initial = FindNearest(original, args.bits, device)

    solver = ALNS(initial, original, X, args.bits, seed=args.seed,
                  acceptance_policy=args.acceptance_policy)
    solver.set_torch_device(device)
    solver.set_LS_operator("S")
    solver.set_stopping_criteria(MaxRuntime(args.seconds))

    synchronize(device)
    started = time.perf_counter()
    solution = solver.solve()
    synchronize(device)
    elapsed = time.perf_counter() - started

    best = solution.alns_result.best_state
    residual = X @ solution.quantized_weights - best.B_k
    statistics = solution.alns_result.statistics

    # Objective against elapsed time, so the value of a longer budget is readable.
    cumulative, trace = 0.0, []
    running = float("inf")
    for runtime, objective in zip(statistics.runtimes, statistics.objectives[1:]):
        cumulative += float(runtime)
        running = min(running, float(objective))
        trace.append((round(cumulative, 3), running))

    def at(seconds):
        reached = [value for moment, value in trace if moment <= seconds]
        return reached[-1] if reached else float(solution.initial_obj)

    report = {
        "instance": name, "row": args.row, "seconds_requested": args.seconds,
        "seconds_elapsed": elapsed, "iterations": len(statistics.runtimes),
        "device": str(device), "acceptance_policy": args.acceptance_policy,
        "initial_objective": float(solution.initial_obj),
        "objective_at": {str(s): at(s) for s in (1, 5, 10, 30, 60, 120, 300)},
        "time_to_best": float(solution.bestFoundSolutionTime),
        "structure": structure(X, original, best.quantization_levels, initial.long(),
                               best.weights, residual),
        "trace": trace,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
