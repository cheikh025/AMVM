"""Does minviol match ALNS/batched.py on real quantization rows, at equal time?

The two implement the same search, so this is not a test of the port -- that is
tests/test_minviol_parity.py, which holds them to an identical trajectory. What
this measures is the one deliberate difference: minviol also runs a
single-variable descent, which the batched engine does not have. Whether that
helps at a fixed budget is an empirical question, and the answer is allowed to be
"no".

Protocol follows experiments/algo_memory.md: equal wall-clock budgets, several
owned seeds, and the two engines alternated within each repetition rather than
measured in blocks. Blocked runs on this machine drift with the power source.

    python experiments/minviol_equal_time.py --seconds 5 --trials 3
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "GPU"))

import torch  # noqa: E402

from ALNS import batched, minviol_engine  # noqa: E402
from utils.utils import select_device  # noqa: E402


def synthetic(n_rows, n_variables, n_samples, bits, seed, device):
    """Log-normal per-channel activation scales, as bench_row_solve.py uses."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(torch.randn(n_variables, generator=g) * 1.5)
    inputs = (torch.randn(n_samples, n_variables, generator=g) * scale).to(device)
    originals = torch.randn(n_rows, n_variables, generator=g).to(device)

    levels_count = 2 ** bits
    low = originals.min(dim=1, keepdim=True).values
    high = originals.max(dim=1, keepdim=True).values
    steps = torch.arange(levels_count, device=device, dtype=originals.dtype)
    levels = low + steps[None, :] * (high - low) / (levels_count - 1)
    initial = (originals[:, :, None] - levels[:, None, :]).abs().argmin(dim=2).long()
    B_k = (inputs @ originals.T).T.contiguous()
    return inputs, initial, levels, B_k


def run(engine, args, device, seed):
    inputs, initial, levels, B_k = synthetic(args.rows, args.variables, args.samples,
                                             args.bits, seed, device)
    started = time.time()
    _, objective = engine.solve(inputs, initial, levels, B_k, args.seconds,
                                acceptance_policy=args.policy, seed=seed)
    return objective.detach().cpu().tolist(), time.time() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--variables", type=int, default=768)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--policy", default="linf_l2_nonincrease")
    parser.add_argument("--output", default="experiments/results/minviol/equal_time.json")
    args = parser.parse_args()

    device = select_device()
    print(f"device={device}  rows={args.rows} variables={args.variables} "
          f"samples={args.samples} bits={args.bits}  {args.seconds}s x {args.trials} trials")

    records = []
    for trial in range(args.trials):
        seed = 9101 + trial
        order = [("batched", batched), ("minviol", minviol_engine)]
        if trial % 2:
            order.reverse()          # alternate, so neither always runs first
        for name, engine in order:
            objectives, elapsed = run(engine, args, device, seed)
            records.append({"engine": name, "seed": seed, "trial": trial,
                            "objectives": objectives, "seconds": elapsed})
            print(f"  trial {trial} {name:8} median {statistics.median(objectives):.6f} "
                  f"in {elapsed:.1f}s")

    print()
    paired_wins = paired_losses = 0
    for trial in range(args.trials):
        a = next(r for r in records if r["trial"] == trial and r["engine"] == "batched")
        b = next(r for r in records if r["trial"] == trial and r["engine"] == "minviol")
        for old, new in zip(a["objectives"], b["objectives"]):
            if new < old - 1e-9:
                paired_wins += 1
            elif new > old + 1e-9:
                paired_losses += 1
    total = args.trials * args.rows
    print(f"paired rows: minviol better on {paired_wins}/{total}, "
          f"worse on {paired_losses}/{total}")
    for name in ("batched", "minviol"):
        flat = [v for r in records if r["engine"] == name for v in r["objectives"]]
        print(f"  {name:8} median objective {statistics.median(flat):.6f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump({"device": str(device), "args": vars(args), "records": records},
                  handle, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
