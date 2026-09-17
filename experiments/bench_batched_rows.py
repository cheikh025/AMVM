"""Compare the per-row solver against the batched solver at equal total time.

Production gives every row its own budget, so a matrix of ``R`` rows costs
``R * seconds`` on one device. The batched solver gets that same total for the
whole set at once. Equal total wall time is the only fair comparison: the
batched loop trades per-row search iterations for far more rows in flight, and
the question is whether the objectives it reaches are better or worse.

    experiments/bench_batched_rows.py --instance experiments/data/<layer>.pt \\
        --rows 16 --seconds-per-row 10
"""

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "GPU"))

from alns.stop import MaxRuntime  # noqa: E402

from ALNS import batched, counters, tuning  # noqa: E402
from ALNS.ALNS import ALNS  # noqa: E402
from RoundToNearest import FindNearest  # noqa: E402
from bench_row_solve import load_instance, synchronize  # noqa: E402
from utils.utils import select_device  # noqa: E402


def uniform_levels(rows, bits, device):
    """Per-row uniform domains, matching what the single-row solver derives."""
    minimum = rows.min(dim=1, keepdim=True).values
    maximum = rows.max(dim=1, keepdim=True).values
    steps = torch.arange(2 ** bits, device=device, dtype=rows.dtype)
    return minimum + steps[None, :] * (maximum - minimum) / (2 ** bits - 1)


def run_sequential(X, W, rows, bits, seconds, device, policy, seed):
    objectives, started = [], time.perf_counter()
    for index, row in enumerate(rows):
        random.seed(seed + index)
        np.random.seed(seed + index)
        torch.manual_seed(seed + index)
        original = W[row, :].contiguous()
        _, initial = FindNearest(original, bits, device)
        solver = ALNS(initial, original, X, bits, seed=seed + index, acceptance_policy=policy)
        solver.set_torch_device(device)
        solver.set_LS_operator("S")
        solver.set_stopping_criteria(MaxRuntime(seconds))
        solution = solver.solve()
        exact = (X @ solution.quantized_weights - solution.alns_result.best_state.B_k)
        objectives.append(float(exact.abs().max().item()))
    synchronize(device)
    return objectives, time.perf_counter() - started


def run_batched(X, W, rows, bits, seconds, device, policy, seed, candidate_tile):
    started = time.perf_counter()
    originals = W[rows, :].contiguous()
    levels = uniform_levels(originals, bits, device)
    initial = torch.stack([FindNearest(originals[i], bits, device)[1]
                           for i in range(len(rows))]).long()
    B_k = (X @ originals.T).T.contiguous()

    weights, objective = batched.solve(X, initial, levels, B_k, seconds,
                                       acceptance_policy=policy, seed=seed,
                                       candidate_tile=candidate_tile)
    synchronize(device)
    elapsed = time.perf_counter() - started

    exact = []
    for index in range(len(rows)):
        residual = X @ levels[index][weights[index]] - B_k[index]
        exact.append(float(residual.abs().max().item()))
        assert abs(exact[-1] - float(objective[index])) < 1e-3, "batched objective is not honest"
    return exact, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="synthetic:M=768:R=64")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--seconds-per-row", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--acceptance-policy", default="linf_l2_nonincrease")
    parser.add_argument("--candidate-tile", type=int, default=4096)
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = select_device(args.device)
    name, X, W = load_instance(args.instance, args.n, device)
    rows = list(range(min(args.rows, W.shape[0])))
    total_seconds = args.seconds_per_row * len(rows)

    sequential, sequential_seconds = ([], 0.0)
    if not args.skip_sequential:
        sequential, sequential_seconds = run_sequential(
            X, W, rows, args.bits, args.seconds_per_row, device, args.acceptance_policy,
            args.seed)
        print(f"sequential: {len(rows)} rows in {sequential_seconds:.1f}s, "
              f"median objective {statistics.median(sequential):.6f}")

    counters.reset()
    batch, batch_seconds = run_batched(X, W, rows, args.bits, total_seconds, device,
                                       args.acceptance_policy, args.seed, args.candidate_tile)
    print(f"batched:    {len(rows)} rows in {batch_seconds:.1f}s, "
          f"median objective {statistics.median(batch):.6f}")
    print(f"counters: {counters.snapshot()}")

    report = {"instance": name, "device": str(device), "rows": len(rows),
              "n": int(X.shape[0]), "m": int(X.shape[1]), "bits": args.bits,
              "seconds_per_row": args.seconds_per_row,
              "sequential_objectives": sequential, "sequential_seconds": sequential_seconds,
              "batched_objectives": batch, "batched_seconds": batch_seconds,
              "counters": counters.snapshot(), "flags": tuning.describe(device)}
    if sequential:
        deltas = [b - s for b, s in zip(batch, sequential)]
        report["median_objective_delta"] = statistics.median(deltas)
        report["rows_better_with_batching"] = sum(1 for d in deltas if d < 0)
        report["rows_worse_with_batching"] = sum(1 for d in deltas if d > 0)
        print(f"median objective delta (negative favours batching): "
              f"{report['median_objective_delta']:+.6f}   "
              f"better {report['rows_better_with_batching']}/{len(rows)}")
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
