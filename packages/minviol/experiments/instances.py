"""The benchmark instance set for general constraint systems.

Every instance carries a planted solution, so feasibility is known to be
attainable and a failure to reach it is a statement about the search rather than
about the problem. They start from ``init="zero"`` deliberately: the warm start
solves the easy ones outright, which measures the initializer rather than the
move classes under test.

Shapes are chosen to span what "general constraint system" can mean -- dense and
sparse, inequality and equality and two-sided, well-conditioned and not, random
and structured.
"""

import torch

LEVELS = 4


def _planted(domain, n_variables, generator, device):
    return domain[torch.randint(0, len(domain), (n_variables,), generator=generator)
                  .to(device)]


def _gaussian(m, n, generator, density=1.0):
    A = torch.randn(m, n, generator=generator)
    if density < 1.0:
        A[torch.rand(m, n, generator=generator) >= density] = 0
    return A


def dense_inequality(seed, device, m=20_000, n=128):
    """A x <= u, dense. The plain form of the pitch."""
    g = torch.Generator().manual_seed(seed)
    A = _gaussian(m, n, g).to(device)
    domain = torch.arange(float(LEVELS), device=device)
    x = _planted(domain, n, g, device)
    return dict(A=A, lower=None, upper=A @ x + 0.5, domain=domain)


def sparse_inequality(seed, device, m=50_000, n=256, density=0.01):
    """A x <= u, 1% dense. The regime the sparse backend is built for."""
    g = torch.Generator().manual_seed(seed)
    A = _gaussian(m, n, g, density).to(device)
    domain = torch.arange(float(LEVELS), device=device)
    x = _planted(domain, n, g, device)
    return dict(A=A, lower=None, upper=A @ x + 0.5, domain=domain, sparse=True)


def equality(seed, device, m=20_000, n=128):
    """A x = b. The hardest shape: no slack anywhere to absorb a wrong variable."""
    g = torch.Generator().manual_seed(seed)
    A = _gaussian(m, n, g).to(device)
    domain = torch.arange(float(LEVELS), device=device)
    x = _planted(domain, n, g, device)
    target = A @ x
    return dict(A=A, lower=target, upper=target, domain=domain)


def two_sided(seed, device, m=20_000, n=128, width=0.5):
    """l <= A x <= u, a narrow band around the planted point."""
    g = torch.Generator().manual_seed(seed)
    A = _gaussian(m, n, g).to(device)
    domain = torch.arange(float(LEVELS), device=device)
    x = _planted(domain, n, g, device)
    reached = A @ x
    return dict(A=A, lower=reached - width, upper=reached + width, domain=domain)


def ill_conditioned(seed, device, m=20_000, n=128, correlation=0.95):
    """Columns leaning on a few shared factors, so the system is near rank-deficient."""
    g = torch.Generator().manual_seed(seed)
    rank = max(1, int(n * (1 - correlation)))
    base = torch.randn(m, n, generator=g)
    factors = torch.randn(m, rank, generator=g)
    loadings = torch.randn(rank, n, generator=g)
    A = ((1 - correlation) * base
         + correlation * (factors @ loadings) / rank ** 0.5).to(device)
    domain = torch.arange(float(LEVELS), device=device)
    x = _planted(domain, n, g, device)
    return dict(A=A, lower=None, upper=A @ x + 0.5, domain=domain)


def tomography(seed, device, size=24, angles=14):
    """A structured sparse equality system: every variable lies on one ray per angle.

    Structured rather than random: column supports are exactly uniform and the
    constraints come in correlated groups, which is what a real projector looks
    like and what a random sparse matrix does not capture.
    """
    detectors = int(size * 1.5)
    centre = (size - 1) / 2.0
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    xs, ys = (xs - centre).reshape(-1), (ys - centre).reshape(-1)
    radius = centre * (2 ** 0.5)

    rows = []
    for a in range(angles):
        theta = torch.tensor(a * torch.pi / angles)
        offset = xs * torch.cos(theta) + ys * torch.sin(theta)
        detector = ((offset + radius) / (2 * radius) * (detectors - 1)).round().long()
        rows.append(a * detectors + detector.clamp(0, detectors - 1))
    index = torch.stack([torch.cat(rows),
                         torch.arange(size * size).repeat(angles)])
    A = torch.zeros(angles * detectors, size * size)
    A[index[0], index[1]] = 1.0
    A = A.to(device)

    g = torch.Generator().manual_seed(seed)
    domain = torch.arange(float(LEVELS), device=device)
    x = _planted(domain, size * size, g, device)
    target = A @ x
    return dict(A=A, lower=target, upper=target, domain=domain, sparse=True)


def binary_dense(seed, device, m=20_000, n=128):
    """A x <= u over {0, 1}. Every variable sits at an end of its domain.

    Included because a two-level domain is the worst case for anything that
    proposes a step without checking it can be taken: half the variables cannot
    move in a given direction at all.
    """
    g = torch.Generator().manual_seed(seed)
    A = _gaussian(m, n, g).to(device)
    domain = torch.tensor([0.0, 1.0], device=device)
    x = _planted(domain, n, g, device)
    return dict(A=A, lower=None, upper=A @ x + 0.5, domain=domain)


def binary_sparse(seed, device, m=40_000, n=256, density=0.01):
    """The same, sparse."""
    g = torch.Generator().manual_seed(seed)
    A = _gaussian(m, n, g, density).to(device)
    domain = torch.tensor([0.0, 1.0], device=device)
    x = _planted(domain, n, g, device)
    return dict(A=A, lower=None, upper=A @ x + 0.5, domain=domain, sparse=True)


SET = {
    "binary-dense": binary_dense,
    "binary-sparse": binary_sparse,
    "dense-ineq": dense_inequality,
    "sparse-ineq": sparse_inequality,
    "equality": equality,
    "two-sided": two_sided,
    "ill-cond": ill_conditioned,
    "tomography": tomography,
}
