"""Where does the sparse backend start to beat the dense one?

The README is not allowed to claim a sparse speedup without this. The answer
depends on the density and on the constraint count, and it is not obvious in
either direction: sparse reads fewer constraints per candidate, but it pays for
ragged gathers and a merge that dense does not need.

Method: the two backends are alternated inside each repetition rather than
measured in blocks. Blocked runs on this machine drift with the power source and
overstated a comparison by 6-8%, and alternating cancels any drift that is slow
compared with one repetition. Both backends score the identical candidate list
from the identical point, so this times the kernels and nothing else.

    python benchmarks/density_crossover.py --constraints 20000
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


def build(n_constraints, n_variables, density, levels, seed, device):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_constraints, n_variables, generator=g)
    keep = torch.rand(n_constraints, n_variables, generator=g) < density
    A = (A * keep).to(device)
    domain = torch.linspace(-1, 1, levels)[None, :].to(device)
    x0 = torch.randint(0, levels, (1, n_variables), generator=g).to(device)
    planted = torch.randint(0, levels, (1, n_variables), generator=g).to(device)
    target = A @ torch.gather(domain, 1, planted).T
    return A, domain, x0, target


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_scoring(batch, options, tag, left, right, delta, screen_rows, device):
    counters = Counters()
    sync(device)
    started = time.perf_counter()
    batch.matrix.exact_scores(batch, tag, left, right, delta, batch.objective,
                              screen_rows, options, counters)
    sync(device)
    return time.perf_counter() - started


def measure(n_constraints, n_variables, density, levels, repeats, seed, device):
    A, domain, x0, target = build(n_constraints, n_variables, density, levels, seed,
                                  device)
    sparse = SparseMatrix.from_dense(A)
    options = Options(prune_min_constraints=0, n_filters=max(128, sparse.max_nnz + 1))
    dense_batch = Batch(DenseMatrix(A), x0, domain, target, target)
    sparse_batch = Batch(sparse, x0, domain, target, target)

    active = torch.ones(1, dtype=torch.bool, device=device)
    rows_d = engine.screening_rows(dense_batch, options.n_filters)
    rows_s = engine.screening_rows(sparse_batch, options.n_filters)
    tag, left, level = engine.candidate_moves(dense_batch, active)
    delta = (dense_batch.domain[tag, level]
             - dense_batch.domain[tag, dense_batch.x_idx[tag, left]])

    dense_times, sparse_times = [], []
    for r in range(repeats + 1):
        # Alternate the order too, so neither backend always runs warm.
        first_dense = r % 2 == 0
        order = [("dense", dense_batch, rows_d), ("sparse", sparse_batch, rows_s)]
        if not first_dense:
            order.reverse()
        taken = {}
        for name, batch, rows in order:
            taken[name] = time_scoring(batch, options, tag, left, None, delta, rows,
                                       device)
        if r == 0:
            continue                       # first repetition warms the kernels
        dense_times.append(taken["dense"])
        sparse_times.append(taken["sparse"])

    return {
        "constraints": n_constraints, "variables": n_variables,
        "density": density, "nnz": sparse.nnz, "max_nnz": sparse.max_nnz,
        "candidates": int(len(tag)),
        "dense_ms": statistics.median(dense_times) * 1e3,
        "sparse_ms": statistics.median(sparse_times) * 1e3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", type=int, nargs="+",
                        default=[2000, 20000, 100000])
    parser.add_argument("--variables", type=int, default=256)
    parser.add_argument("--densities", type=float, nargs="+",
                        default=[0.002, 0.01, 0.05, 0.2, 0.5])
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"device={device}  variables={args.variables}  levels={args.levels}  "
          f"repeats={args.repeats}")
    print(f"{'constraints':>11} {'density':>8} {'nnz/col':>8} {'dense ms':>10} "
          f"{'sparse ms':>10} {'speedup':>8}")
    records = []
    for m in args.constraints:
        for d in args.densities:
            record = measure(m, args.variables, d, args.levels, args.repeats,
                             args.seed, device)
            record["speedup"] = record["dense_ms"] / record["sparse_ms"]
            records.append(record)
            print(f"{record['constraints']:>11} {record['density']:>8.3f} "
                  f"{record['nnz'] / record['variables']:>8.1f} "
                  f"{record['dense_ms']:>10.3f} {record['sparse_ms']:>10.3f} "
                  f"{record['speedup']:>7.2f}x")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"device": str(device), "records": records}, handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
