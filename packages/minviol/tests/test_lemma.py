"""The two claims the sparse shortcut rests on, checked by brute force.

Neither is obvious enough to leave as a comment. The first says a bounded set of
the worst constraints is enough to know the largest violation among the ones a
candidate did not touch. The second says a candidate that misses every currently
worst constraint cannot improve at all, which is what lets the search build its
candidate list from the active constraints instead of from every variable.
"""
import torch

from minviol import violation


def random_system(seed, m=400, n=40, cutoff=1.4, levels=4):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, n, generator=g)
    A[A.abs() < cutoff] = 0
    lower = torch.randn(m, generator=g)
    upper = lower + torch.rand(m, generator=g) * 2.0
    domain = torch.linspace(-1, 1, levels)
    x = domain[torch.randint(0, levels, (n,), generator=g)]
    return A, lower, upper, domain, x


def test_top_k_holds_the_untouched_maximum():
    """max over untouched == max over (top-K minus touched), when that is non-empty."""
    checked = 0
    for seed in range(40):
        A, lower, upper, _, x = random_system(seed)
        v = violation(A @ x, lower, upper)
        top = v.topk(12).indices
        for j in range(A.shape[1]):
            support = A[:, j] != 0
            untouched = ~support
            kept = top[untouched[top]]
            if not untouched.any() or not len(kept):
                continue
            checked += 1
            assert v[untouched].max() == v[kept].max()
    assert checked > 1200, f"only {checked} cases exercised the claim"


def test_tau_bounds_everything_outside_the_kept_set():
    """The (K+1)-th violation bounds every constraint the kept set leaves out."""
    for seed in range(20):
        A, lower, upper, _, x = random_system(seed)
        v = violation(A @ x, lower, upper)
        k = 12
        top = v.topk(k + 1)
        tau = top.values[k]
        outside = torch.ones_like(v, dtype=torch.bool)
        outside[top.indices[:k]] = False
        assert bool((v[outside] <= tau).all())


def test_only_variables_in_an_active_constraint_can_improve():
    """A candidate missing every worst constraint leaves the objective where it was."""
    checked = 0
    for seed in range(25):
        A, lower, upper, domain, x = random_system(seed)
        v = violation(A @ x, lower, upper)
        objective = v.max()
        worst = (v >= objective - 1e-12)
        for j in range(A.shape[1]):
            support = A[:, j] != 0
            covers_every_worst = bool(support[worst].all())
            for level in domain:
                if level == x[j]:
                    continue
                moved = x.clone()
                moved[j] = level
                improved = bool(violation(A @ moved, lower, upper).max()
                                < objective - 1e-9)
                checked += 1
                assert not (improved and not covers_every_worst), (
                    f"seed {seed}: variable {j} improved without touching every "
                    "worst constraint")
    assert checked > 2000, f"only {checked} cases exercised the claim"
