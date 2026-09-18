"""Benchmark and equivalence harness for one-row ALNS solves.

Three modes cover the two verification regimes the solver needs.

``trajectory``
    Fixed seed and a fixed iteration count. Records the full objective sequence
    and a digest of the final solution. Two runs that agree here are exactly
    equivalent, which is the bar for changes that must not alter the search.

``speed``
    Fixed iteration count, wall time only. Measures work per iteration. Never
    run it under a profiler; profiler overhead scales with operation count and
    biases exactly the comparisons this mode exists to make.

``quality``
    Fixed wall-clock budget, the production stopping rule. Records the final
    infinity norm. A per-iteration speedup shows up here as a better objective
    at equal time, which is the only measure that matters in production.

Variants are selected by the environment flags in ``ALNS.tuning`` and compared
with the ``compare`` subcommand, which pairs records by instance, row and seed.
"""

import argparse
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "GPU"))

from alns.stop import MaxIterations, MaxRuntime  # noqa: E402

from ALNS import counters, tuning  # noqa: E402
from ALNS.ALNS import ALNS  # noqa: E402
from RoundToNearest import FindNearest  # noqa: E402
from utils.utils import select_device  # noqa: E402


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_instance(spec, n_rows, device):
    """Return (name, X, W) for a captured instance or a synthetic stand-in.

    ``synthetic`` draws per-channel scales from a log-normal distribution so the
    activation matrix has the heavy-tailed channel structure that real
    transformer activations show. It is a stand-in for a captured layer, not a
    substitute: report captured-layer results whenever they are available.
    """
    if spec.startswith("synthetic"):
        parts = dict(part.split("=") for part in spec.split(":")[1:]) if ":" in spec else {}
        columns = int(parts.get("M", 768))
        rows = int(parts.get("R", 32))
        generator = torch.Generator().manual_seed(int(parts.get("seed", 9101)))
        scales = torch.exp(1.2 * torch.randn(columns, generator=generator))
        X = torch.randn(n_rows, columns, generator=generator) * scales
        W = 0.02 * torch.randn(rows, columns, generator=generator)
        return spec, X.to(device), W.to(device)

    payload = torch.load(spec, map_location="cpu", weights_only=False)
    X, W = payload["X"], payload["W"]
    if n_rows and n_rows < X.shape[0]:
        X = X[:n_rows]
    return payload.get("name", pathlib.Path(spec).stem), X.to(device), W.to(device)


def digest(tensor):
    return hashlib.sha256(np.ascontiguousarray(
        tensor.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def solve_one(X, W, row, bits, seed, device, stopping, acceptance_policy):
    """Run one ALNS row solve exactly as ``executeALNS`` does and time it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Counters are process-local and additive. One process now runs several
    # variants, so they must be cleared per solve or a record would carry the
    # previous variant's counts.
    counters.reset()

    original_row = W[row, :].contiguous()
    _, initial = FindNearest(original_row, bits, device)

    solver = ALNS(initial, original_row, X, bits, seed=seed,
                  acceptance_policy=acceptance_policy)
    solver.set_torch_device(device)
    solver.set_LS_operator("S")
    solver.set_stopping_criteria(stopping)

    synchronize(device)
    started = time.perf_counter()
    solution = solver.solve()
    synchronize(device)
    elapsed = time.perf_counter() - started

    statistics_ = solution.alns_result.statistics
    exact = (X @ solution.quantized_weights - solution.alns_result.best_state.B_k)
    return {
        "row": int(row),
        "seed": int(seed),
        "seconds": elapsed,
        "iterations": len(statistics_.runtimes),
        "initial_objective": float(solution.initial_obj),
        "final_objective": float(solution.inf_norm_value),
        "exact_objective": float(exact.abs().max().item()),
        "final_l2": float(solution.alns_result.best_state.L2_norm.item()),
        "objectives": [float(value) for value in statistics_.objectives],
        "solution_digest": digest(solution.best_state.weights),
        "time_to_best": float(solution.bestFoundSolutionTime),
        "counters": counters.snapshot(),
    }


# Variant keys that are solver arguments rather than tuning flags.
SOLVER_KEYS = {"acceptance_policy"}


def parse_variant(spec):
    """Parse ``LABEL KEY=VALUE ...`` into (label, tuning settings, solver arguments).

    Keys are tuning attributes, except the few in ``SOLVER_KEYS`` which are
    arguments of the solve itself. Both belong in a variant so an experiment over
    either can be interleaved.
    """
    label, settings, solver = spec[0], {}, {}
    for assignment in spec[1:]:
        key, _, raw = assignment.partition("=")
        if not _:
            raise SystemExit(f"variant setting must be KEY=VALUE, got {assignment!r}")
        key = key.strip()
        if key in SOLVER_KEYS:
            solver[key] = raw
            continue
        if not hasattr(tuning, key):
            raise SystemExit(f"unknown tuning attribute {key!r}")
        current = getattr(tuning, key)
        if isinstance(current, bool):
            value = raw.strip().lower() in tuning.TRUTHY
        elif isinstance(current, int):
            value = int(raw)
        else:
            raise SystemExit(f"{key!r} is not a settable flag")
        settings[key] = value
    return label, settings, solver


def apply_variant(settings, baseline):
    """Set the variant's flags, restoring every other flag to the baseline."""
    for key, value in baseline.items():
        setattr(tuning, key, value)
    for key, value in settings.items():
        setattr(tuning, key, value)


def run(args):
    device = select_device(args.device)
    name, X, W = load_instance(args.instance, args.n, device)
    rows = args.rows if args.rows else list(range(min(4, W.shape[0])))

    # One variant by default; several are measured interleaved. Interleaving costs
    # no extra solves, it only reorders them, so that every (row, seed) point is
    # measured for all variants within seconds of each other. Anything that drifts
    # over the run then affects all variants equally instead of whichever one
    # happened to be running.
    variants = ([parse_variant(spec) for spec in args.variant] if args.variant
                else [(args.label, {}, {})])
    settable = [key for key, value in vars(tuning).items()
                if key.isupper() and isinstance(value, (bool, int))]
    baseline = {key: getattr(tuning, key) for key in settable}

    records = {label: [] for label, _, _ in variants}
    point = 0
    for row in rows:
        for seed in args.seeds:
            # Rotate which variant goes first, so no variant always pays whatever
            # the first solve at a point costs.
            order = variants[point % len(variants):] + variants[:point % len(variants)]
            point += 1
            for label, settings, solver in order:
                apply_variant(settings, baseline)
                policy = solver.get("acceptance_policy", args.acceptance_policy)
                stopping = (MaxRuntime(args.seconds) if args.mode == "quality"
                            else MaxIterations(args.iterations))
                record = solve_one(X, W, row, args.bits, seed, device, stopping,
                                   policy)
                record.update(mode=args.mode, instance=name, instance_spec=args.instance,
                              n=int(X.shape[0]), m=int(X.shape[1]), bits=args.bits,
                              device=str(device), label=label,
                              acceptance_policy=policy,
                              flags=tuning.describe(device))
                if args.mode != "trajectory":
                    record.pop("objectives")
                records[label].append(record)
                print(f"[{label}] row {row} seed {seed}: "
                      f"obj {record['final_objective']:.6f} "
                      f"({record['iterations']} iters, {record['seconds']:.2f}s, "
                      f"{record['seconds'] / max(record['iterations'], 1) * 1000:.1f} ms/iter)")
                if abs(record["final_objective"] - record["exact_objective"]) > 1e-4:
                    raise SystemExit(f"cached objective disagrees with direct recomputation: {record}")
    apply_variant({}, baseline)

    # One report per variant, at the layout the summarizer and ``compare`` expect,
    # so an interleaved run is read with the same tools as a single-variant one.
    for label, settings, solver in variants:
        apply_variant(settings, baseline)
        report = {"label": label, "mode": args.mode, "device": str(device),
                  "instance": name, "records": records[label],
                  "flags": tuning.describe(device),
                  "solver_arguments": solver,
                  "interleaved_with": [other for other, _, _ in variants if other != label],
                  "torch": torch.__version__}
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.results_dir:
            # Name the file after the instance file, as run_protocol.sh does, so
            # both protocols write the same layout.
            stem = (pathlib.Path(args.instance).stem
                    if pathlib.Path(args.instance).exists() else name)
            destination = (pathlib.Path(args.results_dir) / label
                           / f"{stem}_{args.mode}.json")
        elif args.out:
            destination = pathlib.Path(args.out)
        else:
            print(rendered)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {destination}")
    apply_variant({}, baseline)


def compare(args):
    """Pair two reports by (row, seed) and summarize the difference."""
    baseline = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))
    variant = json.loads(pathlib.Path(args.variant).read_text(encoding="utf-8"))
    index = {(record["row"], record["seed"]): record for record in variant["records"]}

    identical_trajectories = True
    objective_deltas, speedups, largest_objective_gap = [], [], 0.0
    for base in baseline["records"]:
        other = index.get((base["row"], base["seed"]))
        if other is None:
            raise SystemExit(f"variant is missing row {base['row']} seed {base['seed']}")
        if "objectives" in base and "objectives" in other:
            same_length = len(base["objectives"]) == len(other["objectives"])
            gap = (max(abs(a - b) for a, b in zip(base["objectives"], other["objectives"]))
                   if same_length else float("inf"))
            largest_objective_gap = max(largest_objective_gap, gap)
            if not same_length or gap > args.tolerance or \
                    base["solution_digest"] != other["solution_digest"]:
                identical_trajectories = False
        objective_deltas.append(other["final_objective"] - base["final_objective"])
        speedups.append((base["seconds"] / max(base["iterations"], 1))
                        / max(other["seconds"] / max(other["iterations"], 1), 1e-12))

    improved = sum(1 for delta in objective_deltas if delta < 0)
    worsened = sum(1 for delta in objective_deltas if delta > 0)
    summary = {
        "baseline": baseline["label"], "variant": variant["label"],
        "pairs": len(objective_deltas),
        "trajectories_identical": identical_trajectories,
        "largest_objective_gap": largest_objective_gap,
        "median_objective_delta": statistics.median(objective_deltas),
        "mean_objective_delta": statistics.fmean(objective_deltas),
        "pairs_improved": improved, "pairs_worsened": worsened,
        "median_speedup_per_iteration": statistics.median(speedups),
        "baseline_flags": baseline.get("flags"), "variant_flags": variant.get("flags"),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(rendered + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runner = subparsers.add_parser("run")
    runner.add_argument("--instance", default="synthetic:M=768:R=32")
    runner.add_argument("--mode", choices=["trajectory", "speed", "quality"],
                        default="trajectory")
    runner.add_argument("--rows", type=int, nargs="*")
    runner.add_argument("--seeds", type=int, nargs="+", default=[9101])
    runner.add_argument("--n", type=int, default=16384)
    runner.add_argument("--bits", type=int, default=3)
    runner.add_argument("--iterations", type=int, default=20)
    runner.add_argument("--seconds", type=float, default=10.0)
    runner.add_argument("--device", default=None)
    runner.add_argument("--acceptance-policy", default="linf_l2_nonincrease")
    runner.add_argument("--label", default="run")
    runner.add_argument("--out")
    runner.add_argument("--results-dir",
                        help="write <results-dir>/<variant>/<instance>_<mode>.json, "
                             "the layout run_protocol.sh produces")
    runner.add_argument("--variant", action="append", nargs="+", metavar="SPEC",
                        help="LABEL [TUNING_ATTR=VALUE ...]; repeat to measure "
                             "several variants interleaved in one process")
    runner.set_defaults(function=run)

    comparer = subparsers.add_parser("compare")
    comparer.add_argument("baseline")
    comparer.add_argument("variant")
    comparer.add_argument("--tolerance", type=float, default=0.0)
    comparer.add_argument("--out")
    comparer.set_defaults(function=compare)

    args = parser.parse_args()
    torch.set_grad_enabled(False)  # production runs under no_grad
    args.function(args)


if __name__ == "__main__":
    main()
