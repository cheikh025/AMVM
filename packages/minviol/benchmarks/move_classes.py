"""Which move classes earn their budget on a general constraint system?

The engine came from quantization, where a solve starts near round-to-nearest and
the level histogram is already roughly right. There a swap -- exchanging the
levels of two variables -- is a strong move. A general constraint system starts
nowhere in particular, and this measures whether swaps still pay there.

Equal wall-clock budgets, several seeds, median of the final violation.

    python benchmarks/move_classes.py --constraints 100000
"""

import argparse
import json
import statistics

import torch

import minviol
from minviol import Budget, Options

VARIANTS = {
    "moves+swaps": Options(),
    "moves only": Options(swap_moves=False),
    "swaps only": Options(single_variable_moves=False),
}


def instance(n_constraints, n_variables, density, levels, seed, device):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_constraints, n_variables, generator=g)
    if density < 1.0:
        A[torch.rand(n_constraints, n_variables, generator=g) >= density] = 0
    A = A.to(device)
    domain = torch.arange(float(levels), device=device)
    planted = domain[torch.randint(0, levels, (n_variables,), generator=g).to(device)]
    return A, domain, A @ planted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", type=int, default=100_000)
    parser.add_argument("--variables", type=int, default=256)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.002, 0.01])
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"device={device}  {args.constraints:,} constraints x {args.variables} "
          f"variables, {args.levels} levels, {args.seconds}s x {args.trials} trials")
    print(f"{'density':>8} {'variant':>12} {'backend':>8} {'median violation':>17} "
          f"{'feasible':>9}")

    records = []
    for density in args.densities:
        for name, options in VARIANTS.items():
            for backend in ("sparse", "dense"):
                violations, feasible = [], 0
                for seed in range(args.trials):
                    A, domain, target = instance(args.constraints, args.variables,
                                                 density, args.levels, seed, device)
                    matrix = A.to_sparse_coo() if backend == "sparse" else A
                    result = minviol.solve(matrix, target, target, domain=domain,
                                           init="zero", options=options,
                                           budget=Budget(seconds=args.seconds))
                    violations.append(result.max_violation)
                    feasible += int(result.feasible)
                median = statistics.median(violations)
                records.append({"density": density, "variant": name,
                                "backend": backend, "violations": violations,
                                "median": median, "feasible": feasible})
                print(f"{density:>8.3f} {name:>12} {backend:>8} {median:>17.4f} "
                      f"{feasible:>6}/{args.trials}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"device": str(device), "args": vars(args), "records": records},
                      handle, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
