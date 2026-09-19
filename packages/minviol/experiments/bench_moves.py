"""Canonical evaluation harness for move-class experiments.

Runs named variants over the instance set at equal wall-clock time and reports
the paired comparison. Variants are alternated within each repetition rather than
run in blocks, so slow drift in the machine's state cancels instead of landing on
whichever variant ran second.

    python experiments/bench_moves.py --variants moves-only active-rows
    python experiments/bench_moves.py --variants moves-only --seconds 3 --trials 3

The decision metric is the median paired change in the final violation at equal
time, with the number of pairs that got worse reported alongside. A variant that
improves the median while making some instance worse is not automatically a win.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import minviol                                     # noqa: E402
from minviol import Budget, Options                # noqa: E402

import instances                                   # noqa: E402

# The search as it stood before any experiment in the ledger: a uniformly random
# kick, strict-improvement acceptance, no escalation. Experiments 1, 2 and 4 won
# and became package defaults, so a variant that overrides only the move classes
# now *inherits* those wins -- `moves-only` and `best` are byte-identical, and the
# documented baseline command reports 24 of 24 tied. Pinning the old settings
# explicitly is what keeps the ledger's baseline column reproducible; see the
# "Stale baseline labels" note in experiments/algo_memory.md.
PRE_EXPERIMENT = {"perturbation": "random", "acceptance": "linf",
                  "kick_escalation": False}

# Named variants. Each is a set of Options overrides applied to the package
# defaults. "moves-only" is the incumbent for general constraint systems: swaps
# were measured to be both slower and worse there.
VARIANTS = {
    "default": {},
    "moves+swaps": {"swap_moves": True, "single_variable_moves": True},
    "moves-only": {"swap_moves": False, "single_variable_moves": True},
    "swaps-only": {"swap_moves": True, "single_variable_moves": False},
    # Deliberate duplicate of moves-only. The solver is seeded, so the only
    # run-to-run variance is how many iterations fit in the wall-clock budget;
    # running an identical variant against itself is how that is measured.
    "moves-only-copy": {"swap_moves": False, "single_variable_moves": True},

    # The two historical baselines, pinned. `original-moves-only` is the column
    # the ledger's baseline table calls "moves-only (original)";
    # `original-search` is the swap-based search the whole package replaced.
    "original-moves-only": {"swap_moves": False, "single_variable_moves": True,
                            **PRE_EXPERIMENT},
    "original-search": {"swap_moves": True, "single_variable_moves": True,
                        **PRE_EXPERIMENT},

    # Experiment 1: kick the variables of the worst constraint, in the direction
    # that relieves it, instead of a uniformly random subset in a random direction.
    "active-kick": {"swap_moves": False, "single_variable_moves": True,
                    "perturbation": "active"},
    # The incumbent after experiment 1. Later experiments are measured against
    # this, not against the original moves-only.
    "best": {"swap_moves": False, "single_variable_moves": True,
             "perturbation": "active", "acceptance": "linf_l2_tiebreak",
             "kick_escalation": True},

    # Experiment 4: widen the kick while an instance stalls, snap back on success.
    "escalate": {"swap_moves": False, "single_variable_moves": True,
                 "perturbation": "active", "acceptance": "linf_l2_tiebreak",
                 "kick_escalation": True},


    # Experiment 6: move two variables at once, with independent steps, when
    # nothing else can move the instance.
    "compound": {"swap_moves": False, "single_variable_moves": True,
                 "perturbation": "active", "acceptance": "linf_l2_tiebreak",
                 "kick_escalation": True, "compound_moves": True},
    "compound-wide": {"swap_moves": False, "single_variable_moves": True,
                      "perturbation": "active", "acceptance": "linf_l2_tiebreak",
                      "kick_escalation": True, "compound_moves": True,
                      "compound_width": 128},


    # Experiment 4, refined: escalate only after a long run of failures, so an
    # instance still making progress is left alone.
    "escalate-patient": {"swap_moves": False, "single_variable_moves": True,
                         "perturbation": "active", "acceptance": "linf_l2_tiebreak",
                         "kick_escalation": True, "kick_patience": 20},

    # Experiment 2: let a candidate that leaves the maximum unchanged win on the
    # sum of squares, so the search can cross the plateau a maximum creates.
    "tiebreak": {"swap_moves": False, "single_variable_moves": True,
                 "perturbation": "active", "acceptance": "linf_l2_tiebreak"},

    # Experiment 8: settle ties on a higher power of the violation. The sum stays
    # over every constraint -- a fixed set, so still a potential function -- while
    # a larger power concentrates it on the worst constraints, approaching the
    # lexicographic order on the sorted violation vector. `power2` is a pinned
    # duplicate of `best`, as the control.
    "power2": {"swap_moves": False, "single_variable_moves": True,
               "perturbation": "active", "acceptance": "linf_l2_tiebreak",
               "kick_escalation": True, "tiebreak_power": 2.0},
    "power4": {"swap_moves": False, "single_variable_moves": True,
               "perturbation": "active", "acceptance": "linf_l2_tiebreak",
               "kick_escalation": True, "tiebreak_power": 4.0},
    "power8": {"swap_moves": False, "single_variable_moves": True,
               "perturbation": "active", "acceptance": "linf_l2_tiebreak",
               "kick_escalation": True, "tiebreak_power": 8.0},
    "power16": {"swap_moves": False, "single_variable_moves": True,
                "perturbation": "active", "acceptance": "linf_l2_tiebreak",
                "kick_escalation": True, "tiebreak_power": 16.0},
}


def register(name, **overrides):
    """Add a variant, so an experiment can define its own without editing this list."""
    VARIANTS[name] = overrides


def options_for(name):
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return replace(Options(prune_min_constraints=0), **VARIANTS[name])


def release(device):
    """Hand the device's cached blocks back before the next solve.

    Without this the second variant on an instance inherits the first one's
    memory pressure, and the comparison stops being about the algorithm. It was
    measured: one 3-second solve took 419 seconds under accumulated pressure and
    3.0 seconds in isolation, reaching the same answer either way. Alternating
    the variants does not cancel this, because the cost lands on whichever runs
    second.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def run_one(instance, options, seconds, device, iterations=None):
    """Solve once. With ``iterations`` set the budget is deterministic.

    Wall-clock budgets make a move-class comparison irreproducible on instances
    whose outcome is bimodal -- they either find a feasible point or stall near
    20, and which happens depends on how many iterations happened to fit. The
    same variant on the same instance gave 22.89, then 0.00, then 14.44 across
    three runs. Counting iterations removes that entirely; wall time is still
    reported, so a variant that wins per iteration by being slower is still
    caught.
    """
    matrix = (instance["A"].to_sparse_coo() if instance.get("sparse")
              else instance["A"])
    release(device)
    budget = (Budget(seconds=seconds) if iterations is None
              else Budget(seconds=10_000.0, max_iterations=iterations))
    started = time.time()
    result = minviol.solve(matrix, instance["lower"], instance["upper"],
                           domain=instance["domain"], init="zero", options=options,
                           budget=budget)
    wall = time.time() - started
    del matrix
    release(device)
    return {"violation": result.max_violation, "feasible": result.feasible,
            "iterations": result.iterations, "wall": wall,
            "counters": result.counters}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["moves-only"])
    parser.add_argument("--instances", nargs="+", default=sorted(instances.SET))
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--iterations", type=int, default=None,
                        help="deterministic budget: run exactly this many "
                             "perturb-and-descend rounds instead of using the clock")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")

    budget_note = (f"{args.iterations} iterations" if args.iterations
                   else f"{args.seconds}s")
    print(f"device={device}  {budget_note} x {args.trials} trials  "
          f"variants={args.variants}")
    records = []
    for name in args.instances:
        build = instances.SET[name]
        for trial in range(args.trials):
            instance = build(seed=trial, device=device)
            order = list(args.variants)
            if trial % 2:
                order.reverse()          # alternate, so neither always runs first
            for variant in order:
                outcome = run_one(instance, options_for(variant), args.seconds,
                                  device, args.iterations)
                records.append({"instance": name, "variant": variant, "trial": trial,
                                **outcome})
            del instance
            release(device)
        report(records, name, args.variants)

        limit = args.seconds * 3 + 1 if args.iterations is None else float("inf")
        overran = [r for r in records if r["instance"] == name
                   and r["wall"] > limit]
        if overran:
            # Loudly, because a solve that ignored its budget makes every
            # equal-time comparison in this run meaningless.
            worst = max(r["wall"] for r in overran)
            print(f"    WARNING: {len(overran)} solve(s) overran the "
                  f"{args.seconds}s budget, worst {worst:.0f}s")

    print()
    summarise(records, args.variants, args.trials)

    output = args.output or (f"experiments/results/"
                             f"{'_vs_'.join(args.variants)}.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump({"device": str(device), "args": vars(args), "records": records},
                  handle, indent=2, default=str)
    print(f"wrote {output}")


def report(records, instance, variants):
    line = f"  {instance:14}"
    for variant in variants:
        rows = [r for r in records
                if r["instance"] == instance and r["variant"] == variant]
        vals = [r["violation"] for r in rows]
        feas = sum(r["feasible"] for r in rows)
        wall = statistics.median(r["wall"] for r in rows)
        line += (f"  {variant}={statistics.median(vals):9.4f} "
                 f"(feas {feas}/{len(vals)}, {wall:5.1f}s)")
    print(line)


def summarise(records, variants, trials):
    if len(variants) < 2:
        return
    base, new = variants[0], variants[1]
    better = worse = same = 0
    ratios = []
    for instance in sorted({r["instance"] for r in records}):
        for trial in range(trials):
            def pick(variant):
                return next(r["violation"] for r in records
                            if r["instance"] == instance and r["variant"] == variant
                            and r["trial"] == trial)
            a, b = pick(base), pick(new)
            if b < a - 1e-9:
                better += 1
            elif b > a + 1e-9:
                worse += 1
            else:
                same += 1
            ratios.append(b / a if a > 0 else (1.0 if b == 0 else float("inf")))
    total = better + worse + same
    print(f"paired over {total}: {new} better on {better}, worse on {worse}, "
          f"tied on {same}")
    finite = [r for r in ratios if r != float("inf")]
    if finite:
        print(f"median violation ratio ({new} / {base}): {statistics.median(finite):.4f}")


if __name__ == "__main__":
    main()
