"""Find an integer point satisfying a large system of inequalities.

The headline case: many constraints, few variables, a small integer domain, and
the only question is whether something feasible can be found quickly. The system
is built with a planted solution so that feasibility is known to be attainable --
otherwise a failure to find one would be uninformative.

    python examples/feasibility.py --constraints 200000 --variables 512
    python examples/feasibility.py --correlation 0.98      # much harder
"""

import argparse
import time

import torch

import minviol


def build(n_constraints, n_variables, n_levels, slack, correlation, seed, device):
    """A random system with a planted feasible point.

    ``correlation`` leans the columns on a smaller set of shared factors. At zero
    the columns are independent and the least-squares start usually lands on the
    planted point outright; near one the matrix is close to rank-deficient, the
    start is far off, and the search has to do the work. Both are worth seeing.
    """
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(n_constraints, n_variables, generator=generator)
    if correlation > 0:
        rank = max(1, int(n_variables * (1 - correlation)))
        factors = torch.randn(n_constraints, rank, generator=generator)
        loadings = torch.randn(rank, n_variables, generator=generator)
        A = (1 - correlation) * A + correlation * (factors @ loadings) / rank ** 0.5
    A = A.to(device)

    domain = torch.arange(n_levels, dtype=torch.float32, device=device)
    planted = domain[torch.randint(0, n_levels, (n_variables,), generator=generator)
                     .to(device)]
    # One-sided constraints: A x <= u, with the planted point strictly inside.
    return A, domain, planted, A @ planted + slack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", type=int, default=100_000)
    parser.add_argument("--variables", type=int, default=256)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--slack", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--correlation", type=float, default=0.0,
                        help="0 is independent columns; near 1 is nearly rank-deficient "
                             "and much harder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    A, domain, planted, upper = build(args.constraints, args.variables, args.levels,
                                      args.slack, args.correlation, args.seed, device)
    lower = torch.full_like(upper, float("-inf"))
    print(f"device={device}  {args.constraints:,} constraints x {args.variables} "
          f"variables, {args.levels} levels")
    print(f"matrix: {A.element_size() * A.nelement() / 2**20:.0f} MiB dense")

    # What the starting point alone achieves, for comparison. A zero budget runs
    # the initialization and stops.
    start = minviol.solve(A, None, upper, domain=domain,
                          budget=minviol.Budget(seconds=0.0))
    print(f"starting point violates by {start.max_violation:.4g}")

    started = time.time()
    result = minviol.solve(A, None, upper, domain=domain,
                           budget=minviol.Budget(seconds=args.seconds))
    elapsed = time.time() - started

    print(f"\n{result}")
    if result.feasible:
        print(f"found a feasible point in {result.wall_time:.2f}s "
              f"of a {args.seconds:.0f}s budget")
    else:
        print(f"no feasible point found; worst constraint is off by "
              f"{result.max_violation:.4g}")

    # Independent check: recompute from the returned point, with none of the
    # solver's cached state.
    recomputed = minviol.violation(A @ result.x, lower, upper).max()
    print(f"violation recomputed from scratch: {float(recomputed):.6g}")
    print(f"point differs from the planted one on "
          f"{int((result.x != planted).sum())}/{args.variables} variables "
          f"-- any feasible point will do")
    print(f"{elapsed:.1f}s total")


if __name__ == "__main__":
    main()
