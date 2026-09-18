"""Do the solver changes help the other two applications, or only quantization?

Quantization, tomography and FIR design share one solver but not one problem
shape. The change that mattered most for quantization, visiting residual rows in
descending magnitude inside the pruned exact stage, works because the objective
there is attained at a single sample out of 16,384: put that sample first and a
candidate's partial maximum is nearly final after one stage, so the pruning bound
drops almost everything.

That premise is not guaranteed elsewhere. A Chebyshev optimum over m parameters
equioscillates at about m+2 points, so a well-designed FIR filter should have many
samples tied at the maximum, and ordering cannot separate candidates that are all
tied on the same rows. This script measures the active set for each application
next to the speed numbers, so the result explains itself rather than just landing.

Both applications stop on an iteration count rather than a runtime, so the
local-search deadline never arms for them; that is asserted here rather than
assumed.

Nothing in tomograph.py or fir_design.py is modified or imported for its side
effects: the solver is called with the arguments those functions would pass.
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "GPU"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from alns.stop import MaxIterations  # noqa: E402

from ALNS import counters, tuning  # noqa: E402
from ALNS.ALNS import ALNS  # noqa: E402
from bench_row_solve import synchronize  # noqa: E402
from utils.utils import select_device  # noqa: E402


def tomography_instance(size=32, angles=8, bits=2, max_range=255.0, seed=9101):
    """A parallel-beam instance with an explicit sinogram target.

    The projection matrix is scikit-image's parallel-beam projector, applied to
    unit-pixel images so the matrix is explicit. ASTRA's 'linear' projector, which
    the application uses, interpolates differently, so these numbers predict the
    deployed geometry only qualitatively.
    """
    from skimage.transform import iradon, radon

    rng = np.random.default_rng(seed)
    levels = np.linspace(0.0, max_range, 2 ** bits)
    # A blocky phantom: discrete tomography assumes few materials.
    coarse = rng.integers(0, 2 ** bits, size=(4, 4))
    image = levels[np.kron(coarse, np.ones((size // 4, size // 4), dtype=int))]

    theta = np.linspace(0.0, 180.0, angles, endpoint=False)
    basis = np.zeros((size, size))
    columns = []
    for pixel in range(size * size):
        basis.flat[pixel] = 1.0
        columns.append(radon(basis, theta=theta, circle=False).ravel())
        basis.flat[pixel] = 0.0
    A = np.stack(columns, axis=1)

    sinogram = A @ image.ravel()
    # Filtered back-projection stands in for the SART warm start.
    start = iradon(sinogram.reshape(-1, angles), theta=theta, circle=False,
                   filter_name="ramp")[:size, :size].ravel()
    start = np.clip(start, 0.0, max_range)
    return {"name": f"tomography_{size}x{size}_{angles}angles_{bits}bit",
            "A": A.astype(np.float32), "target": sinogram.astype(np.float32),
            "start": start.astype(np.float32),
            "domain": levels.astype(np.float32), "bits": bits,
            "dtype": torch.float32, "use_fir": False, "ls_op": "S"}


def fir_instance():
    """The filter the application's own main block designs."""
    from fir_config import project_specs
    import fir_design

    order = project_specs["order"]
    points = order * project_specs["grid_density"]
    edges = fir_design.get_remez_edges_from_specs(project_specs)
    grid, desired = fir_design.make_fir_desired(points, project_specs["filter_type"],
                                                edges, fs=project_specs["fs"])
    A = fir_design.create_fir_matrix_A(L=order, angular_omega_grid=grid)
    ideal = fir_design.design_ideal_remez_filter(
        order=order, ftype=project_specs["filter_type"], edges=edges,
        fs=project_specs["fs"], K=project_specs["K"])
    start = fir_design.quantize_coefficients_to_S(ideal, project_specs["p_bits"])
    start = np.asarray(start, dtype=np.float64)[:A.shape[1]]
    return {"name": f"fir_order{order}_{project_specs['p_bits']}bit",
            "A": A, "target": np.asarray(desired, dtype=np.float64), "start": start,
            "domain": None, "bits": project_specs["p_bits"],
            "dtype": torch.float64, "use_fir": True, "ls_op": "W"}


def active_set(residual):
    """How many samples decide the maximum: the premise residual ordering needs."""
    peak = residual.abs().max()
    within = lambda f: int((residual.abs() >= f * peak).sum())
    return {"samples": int(residual.numel()), "at_maximum": within(1.0 - 1e-9),
            "within_1pct": within(0.99), "within_10pct": within(0.90)}


def solve(instance, device, iterations, seed):
    """One solve, with the arguments the application would pass."""
    dtype = instance["dtype"]
    if device.type == "mps" and dtype == torch.float64:
        dtype = torch.float32  # MPS has no float64
    A = torch.as_tensor(instance["A"], dtype=dtype, device=device)
    target = torch.as_tensor(instance["target"], dtype=dtype, device=device)
    start = torch.as_tensor(instance["start"], dtype=dtype, device=device)

    if instance["use_fir"]:
        import fir_design
        initial = fir_design.FindNearest(start, instance["bits"], device)[1]
        domain = None
    else:
        domain = torch.as_tensor(instance["domain"], dtype=dtype, device=device)
        initial = (start[:, None] - domain[None, :]).abs().argmin(dim=1)

    solver = ALNS(initial, start, A, instance["bits"], seed=seed,
                  use_fir=instance["use_fir"], discrete_domain=domain, B_k=target)
    solver.set_torch_device(device)
    solver.set_LS_operator(instance["ls_op"])
    solver.set_stopping_criteria(MaxIterations(iterations))

    counters.reset()
    synchronize(device)
    started = time.perf_counter()
    solution = solver.solve()
    synchronize(device)
    elapsed = time.perf_counter() - started

    best = solution.alns_result.best_state
    # An iteration budget carries no deadline, so the descent limit never arms.
    assert best.deadline is None, "an iteration budget must not arm the deadline"
    residual = A @ solution.quantized_weights - target
    statistics_ = solution.alns_result.statistics
    return {"seconds": elapsed, "iterations": len(statistics_.runtimes),
            "initial_objective": float(solution.initial_obj),
            "final_objective": float(solution.inf_norm_value),
            "exact_objective": float(residual.abs().max()),
            "objectives": [float(v) for v in statistics_.objectives],
            "solution_digest": hashlib.sha256(
                np.ascontiguousarray(solution.quantized_weights.detach().cpu().numpy())
                .tobytes()).hexdigest()[:16],
            "active_set": active_set(residual),
            "counters": counters.snapshot()}


VARIANTS = [("new-defaults", {}),
            ("old", {"PRUNE_TO_INCUMBENT": False, "RESIDUAL_ORDERED_ROWS": False})]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tomography-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--only", choices=[label for label, _ in VARIANTS],
                        help="measure one variant only, so a process never mixes two. "
                             "Variants that differ in tensor shapes warm each other's "
                             "compiled kernels, which flatters whichever runs second.")
    parser.add_argument("--out", default="experiments/results/applications/applications.json")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    baseline = {k: getattr(tuning, k) for k in vars(tuning)
                if k.isupper() and isinstance(getattr(tuning, k), (bool, int))}

    # Tomography follows the solver's own device choice; FIR keeps float64, which
    # MPS cannot provide, so it runs where the deployed numerics are reproducible.
    cases = [(tomography_instance(size=size), select_device())
             for size in args.tomography_sizes]
    cases.append((fir_instance(), torch.device("cpu")))

    records = []
    for instance, device in cases:
        print(f"=== {instance['name']} on {device}  "
              f"A={instance['A'].shape}  levels={2 ** instance['bits']}")
        for trial in range(args.trials):
            # Alternate which variant goes first, as the interleaved runner does.
            order = VARIANTS if trial % 2 == 0 else list(reversed(VARIANTS))
            if args.only:
                order = [v for v in VARIANTS if v[0] == args.only]
            for label, settings in order:
                for key, value in baseline.items():
                    setattr(tuning, key, value)
                for key, value in settings.items():
                    setattr(tuning, key, value)
                result = solve(instance, device, args.iterations, 9101 + trial)
                result.update(instance=instance["name"], variant=label, trial=trial,
                              device=str(device))
                records.append(result)
                print(f"  [{label:13s}] trial {trial}: obj {result['final_objective']:.6f}"
                      f"  {result['seconds'] / max(result['iterations'], 1) * 1000:8.2f} ms/iter"
                      f"  active set {result['active_set']['at_maximum']}"
                      f"/{result['active_set']['within_1pct']}"
                      f"/{result['active_set']['samples']}")
    for key, value in baseline.items():
        setattr(tuning, key, value)

    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
