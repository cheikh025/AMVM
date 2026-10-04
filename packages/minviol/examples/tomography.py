"""Discrete tomography: reconstruct an image from a few projections.

A worked example of the shape minviol is for. Each (angle, detector) pair is one
constraint saying "the pixels along this ray sum to this much", each pixel is a
variable drawn from a small set of grey levels, and the projector is sparse: a
pixel lies on exactly one ray per angle.

Sparse is the right backend here by a wide margin. With D detectors the matrix is
about 1/D dense, so at 64 detectors it is under 2%.

    python examples/tomography.py --size 32 --angles 16 --levels 3
"""

import argparse
import time

import torch

import minviol


def projector(size, n_angles, n_detectors, device, dtype=torch.float32):
    """Parallel-beam projection matrix, nearest-detector ray assignment.

    Returns a sparse ``(angles * detectors, size * size)`` matrix. Each pixel
    contributes to exactly one detector per angle, so every column holds exactly
    ``n_angles`` entries -- uniform support, which is the easy case for a
    segmented gather.
    """
    centre = (size - 1) / 2.0
    ys, xs = torch.meshgrid(torch.arange(size, dtype=dtype, device=device),
                            torch.arange(size, dtype=dtype, device=device),
                            indexing="ij")
    xs, ys = (xs - centre).reshape(-1), (ys - centre).reshape(-1)
    pixel = torch.arange(size * size, device=device)

    radius = centre * (2 ** 0.5)
    rows, cols = [], []
    for a in range(n_angles):
        theta = torch.tensor(a * torch.pi / n_angles, dtype=dtype, device=device)
        offset = xs * torch.cos(theta) + ys * torch.sin(theta)
        detector = ((offset + radius) / (2 * radius) * (n_detectors - 1)).round().long()
        detector = detector.clamp(0, n_detectors - 1)
        rows.append(a * n_detectors + detector)
        cols.append(pixel)

    indices = torch.stack([torch.cat(rows), torch.cat(cols)])
    values = torch.ones(indices.shape[1], dtype=dtype, device=device)
    return minviol.SparseMatrix(indices, values, (n_angles * n_detectors, size * size),
                                device=device, dtype=dtype)


def phantom(size, n_levels, device, dtype=torch.float32):
    """Two overlapping discs on a background, quantized to the grey levels."""
    centre = (size - 1) / 2.0
    ys, xs = torch.meshgrid(torch.arange(size, dtype=dtype, device=device),
                            torch.arange(size, dtype=dtype, device=device),
                            indexing="ij")
    image = torch.zeros(size, size, dtype=dtype, device=device)
    image[((xs - centre * 0.7) ** 2 + (ys - centre) ** 2) < (centre * 0.55) ** 2] = 1.0
    image[((xs - centre * 1.3) ** 2 + (ys - centre) ** 2) < (centre * 0.40) ** 2] = 2.0
    return image.clamp(0, n_levels - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--init", default="lstsq_round",
                        choices=["lstsq_round", "zero", "given"])
    parser.add_argument("--detectors", type=int, default=None)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    detectors = args.detectors or int(args.size * 1.5)

    A = projector(args.size, args.angles, detectors, device)
    truth = phantom(args.size, args.levels, device)
    measured = A.matvec(truth.reshape(1, -1))[:, 0]
    domain = torch.arange(args.levels, dtype=torch.float32, device=device)

    density = A.nnz / (A.shape[0] * A.shape[1])
    print(f"device={device}  image={args.size}x{args.size}  angles={args.angles}  "
          f"detectors={detectors}  levels={args.levels}")
    print(f"constraints={A.shape[0]}  variables={A.shape[1]}  "
          f"nnz={A.nnz} ({density:.2%} dense, {A.max_nnz} per column)")

    # The projections are exact measurements, so the constraints are equalities.
    #
    # The starting point matters more here than anything else. Rounding a
    # least-squares fit onto the grey levels reaches 99.5% of pixels on this
    # instance; starting every pixel at zero stalls around 71%, because the
    # descent has to discover the whole image one pixel at a time. Run with
    # --init zero to see it.
    started = time.time()
    result = minviol.solve(A, measured, measured, domain=domain, init=args.init,
                           budget=minviol.Budget(seconds=args.seconds))
    print(f"\n{result}")

    image = result.x.reshape(args.size, args.size)
    correct = float((image == truth).to(torch.float32).mean())
    print(f"pixels matching the phantom: {correct:.1%}")

    # An independent check: the solver's own number must survive recomputation.
    recomputed = minviol.violation(A.matvec(image.reshape(1, -1))[:, 0],
                                   measured, measured).max()
    print(f"violation recomputed from scratch: {float(recomputed):.6g}")

    shades = " .:-=+*#%@"
    print("\ntruth" + " " * (args.size - 3) + "reconstruction")
    for r in range(0, args.size, max(1, args.size // 24)):
        left = "".join(shades[int(v * (len(shades) - 1) / max(1, args.levels - 1))]
                       for v in truth[r].tolist())
        right = "".join(shades[int(v * (len(shades) - 1) / max(1, args.levels - 1))]
                        for v in image[r].tolist())
        print(f"{left}  {right}")
    print(f"\n{time.time() - started:.1f}s total")


if __name__ == "__main__":
    main()
