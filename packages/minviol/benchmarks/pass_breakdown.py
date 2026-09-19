"""How much of the sparse scoring advantage survives a whole search pass?

`density_crossover.py` times candidate scoring in isolation, where the sparse
backend is up to 35x faster. That is not what a caller experiences. A pass also
screens, refreshes caches, recomputes the top-K set and applies a move, and those
are proportional to the constraint count whatever the backend holds.

This produces the second table in the README, split by move class, because the two
behave oppositely: the single-variable pass is a real win and the swap pass is a
loss large enough to sink the total.

Backends are alternated within each repetition rather than measured in blocks.

    python benchmarks/pass_breakdown.py --constraints 100000
"""

import argparse
import json
import statistics
import time

import torch

from minviol import engine
from minviol.backends import DenseMatrix, SparseMatrix
from minviol.counters import Counters
from minviol.options import Options
from minviol.problem import Batch


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build(n_constraints, n_variables, density, levels, seed, device):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_constraints, n_variables, generator=g)
    A[torch.rand(n_constraints, n_variables, generator=g) >= density] = 0
    A = A.to(device)
    domain = torch.linspace(-1, 1, levels)[None, :].to(device)
    x0 = torch.randint(0, levels, (1, n_variables), generator=g).to(device)
    planted = torch.randint(0, levels, (1, n_variables), generator=g).to(device)
    target = A @ torch.gather(domain, 1, planted).T
    return A, domain, x0, target


def measure(n_constraints, n_variables, density, levels, repeats, seed, device):
    A, domain, x0, target = build(n_constraints, n_variables, density, levels, seed,
                                  device)
    sparse = SparseMatrix.from_dense(A)
    dense = DenseMatrix(A)
    options = Options(prune_min_constraints=0, n_filters=max(128, sparse.max_nnz + 1))

    def fresh(matrix):
        batch = Batch(matrix, x0, domain, target, target)
        return batch, torch.ones(1, dtype=torch.bool, device=device)

    def time_it(matrix, what):
        batch, active = fresh(matrix)
        rows = engine.screening_rows(batch, options.n_filters)
        if what == "move":
            call = lambda: engine.move_pass(batch, active, rows, options, Counters())
        elif what == "swap":
            call = lambda: engine.swap_pass(batch, active, rows, options, Counters())
        else:
            call = lambda: engine.local_search(*fresh(matrix), options, Counters(),
                                               max_passes=3)
        call()
        sync(device)
        started = time.perf_counter()
        for _ in range(repeats):
            call()
        sync(device)
        return (time.perf_counter() - started) / repeats * 1e3

    record = {"constraints": n_constraints, "variables": n_variables,
              "density": density, "max_nnz": sparse.max_nnz,
              "screening_set": options.n_filters}
    for what in ("move", "swap", "passes"):
        # Alternate the order per stage so neither backend always runs warm.
        pair = [("dense", dense), ("sparse", sparse)]
        if what == "swap":
            pair.reverse()
        taken = {name: time_it(matrix, what) for name, matrix in pair}
        record[f"{what}_dense_ms"] = taken["dense"]
        record[f"{what}_sparse_ms"] = taken["sparse"]
        record[f"{what}_speedup"] = taken["dense"] / taken["sparse"]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", type=int, default=100_000)
    parser.add_argument("--variables", type=int, default=256)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.002, 0.01, 0.05])
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"device={device}  constraints={args.constraints:,}  "
          f"variables={args.variables}  levels={args.levels}")
    print(f"{'density':>8} {'K':>6} | {'single-variable':>22} | {'swap':>22} | "
          f"{'three passes':>22}")
    records = []
    for density in args.densities:
        r = measure(args.constraints, args.variables, density, args.levels,
                    args.repeats, args.seed, device)
        records.append(r)
        cells = " | ".join(
            f"{r[f'{w}_dense_ms']:7.1f}/{r[f'{w}_sparse_ms']:7.1f} {r[f'{w}_speedup']:5.2f}x"
            for w in ("move", "swap", "passes"))
        print(f"{density:>8.3f} {r['screening_set']:>6} | {cells}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"device": str(device), "records": records}, handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
