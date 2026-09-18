"""Is one long solve better than the best of several short ones?

A row solve is a draw from a wide distribution: the same row and budget with a
different seed lands 2% to 16% apart depending on the layer. That variance is
either a liability or a resource. Splitting one budget across R independent
replicas and keeping the best trades expected search depth for a better draw, and
which way that trades is an empirical question, per layer.

Replicas are run sequentially here rather than through the batched module. The
batched path reached parity, not speedup, so R replicas cost about R times the
work either way, and running them one at a time keeps the comparison to the
question actually being asked: how to spend a fixed budget.

Both arms are repeated over several trials with disjoint seeds, because each arm
is itself a draw and one sample of each would compare nothing.
"""
import argparse
import json
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "GPU"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from alns.stop import MaxRuntime  # noqa: E402

from bench_row_solve import load_instance, solve_one  # noqa: E402
from utils.utils import select_device  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--acceptance-policy", default="linf_l2_nonincrease")
    parser.add_argument("--out", default="experiments/results/replicas/replicas.json")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = select_device()
    records = []
    for spec in args.instances:
        name, X, W = load_instance(spec, args.n, device)
        for row in args.rows:
            for trial in range(args.trials):
                # Disjoint seeds per trial, so no arm reuses another's draw.
                base = 9101 + 1000 * trial + 37 * row
                for replicas in args.replicas:
                    share = args.budget / replicas
                    best, spent = None, 0.0
                    for r in range(replicas):
                        record = solve_one(X, W, row, args.bits, base + r, device,
                                           MaxRuntime(share), args.acceptance_policy)
                        spent += record["seconds"]
                        if best is None or record["final_objective"] < best:
                            best = record["final_objective"]
                    records.append({"instance": name, "row": row, "trial": trial,
                                    "replicas": replicas, "share_seconds": share,
                                    "spent_seconds": spent, "best_objective": best})
                    print(f"[{name}] row {row} trial {trial} R={replicas}: "
                          f"best {best:.6f} in {spent:.1f}s")
    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"budget": args.budget, "records": records},
                                      indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
