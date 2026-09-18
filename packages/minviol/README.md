# minviol

Find a point that satisfies a large system of constraints, by driving the largest
violation down on a GPU.

```python
import torch, minviol

result = minviol.solve(
    A,                                   # (constraints, variables)
    lower, upper,                        # lower <= A x <= upper
    domain=torch.arange(0, 8.0),         # x drawn from these levels
    budget=minviol.Budget(seconds=10.0),
)

result.feasible        # True when every constraint holds
result.x               # the point, guaranteed to be in the domain
result.max_violation   # how far off it is, 0.0 when feasible
```

It answers *"give me something feasible, fast"*. It does not prove optimality and
it does not prove infeasibility: a non-zero answer means **not found**, never
**infeasible**.

## The problem

```
minimize over x, with each x_j drawn from a finite sorted domain

    max_k  violation_k(x),     violation_k = max(0, lower_k - (Ax)_k, (Ax)_k - upper_k)
```

The objective is zero exactly when `x` is feasible. One form covers every linear
row shape:

| Constraint | How to write it |
|---|---|
| `a·x ≤ u` | `lower = -inf`, `upper = u` |
| `a·x ≥ l` | `lower = l`, `upper = +inf` |
| `a·x = b` | `lower = upper = b` |
| `l ≤ a·x ≤ u` | both finite |

### Why there are no slack variables

For `a·x ≤ u` the slack formulation minimizes `|a·x + s − u|` over `s ≥ 0`. For a
fixed `x` the best slack is `s* = max(0, u − a·x)`, and substituting it back gives
exactly `max(0, a·x − u)`. The slack has a closed-form optimum, so it is
eliminated analytically instead of searched over.

That is not only tidier. One slack per constraint would move the constraint count
— up to hundreds of thousands — into the *variable* count, and variables are the
expensive dimension here: constraints are reduced with a single parallel `max`,
while variables drive candidate enumeration. Slacks are also continuous, and
every variable in this solver lives on a finite sorted domain.

## Where it fits

**Reasonable choice when:** the constraint count is large relative to the variable
count; variables are discrete over a small domain; a good-enough point beats a
certificate.

**Wrong tool when:** you need a proof of infeasibility or an optimality bound —
use an LP/MIP solver's phase 1, which is exact and handles continuous variables.
Also wrong when the variable count is what is large: the sparse backend cuts the
work per candidate, not the number of candidates.

It sits in the same family as the feasibility pump — minimize distance to
feasibility rather than optimize an objective — with the difference that the whole
inner loop is a batched tensor program, so hundreds of thousands of constraints
are routine.

## Honest limits

- **Heuristic.** No dual, no certificate, no gap.
- **Discrete variables only.** No continuous variables.
- **One sorted domain per instance**, shared by all of its variables. Per-variable
  domains are a known gap: the swap kernel factorizes as
  `step · (A[:, i] − A[:, j])` precisely because both variables move by the same
  step. Per-variable freedom is available through `fixed`.
- **Small domains.** A swap pass costs one kernel launch per adjacent level pair:
  fine at 8 levels, ruinous at 1000.
- **Dense memory is `O(m·n)`**: 262,144 constraints × 768 variables in float32 is
  about 805 MB. Sparse is `O(nnz)`, carried in two orderings.
- **Default acceptance is `linf`.** The two L2 policies are refused on
  inequalities rather than quietly degraded: the squared positive part is zero
  wherever a one-sided constraint holds, so the tie-break would stop
  discriminating once most rows were satisfied.

## Dense or sparse

`solve` follows the type you hand it — a dense tensor uses the dense backend, a
`torch.sparse` tensor uses the sparse one. It does not guess from density, because
silently densifying a matrix you deliberately kept sparse is a surprising way to
run out of memory. Use this to choose.

Candidate scoring, 256 variables, MPS, backends alternated within each repetition
(`benchmarks/density_crossover.py`):

| Constraints | Density | Dense | Sparse | Sparse speedup |
|---:|---:|---:|---:|---:|
| 20,000 | 0.2% | 18.1 ms | 3.4 ms | **5.3×** |
| 20,000 | 5% | 18.0 ms | 7.2 ms | **2.5×** |
| 20,000 | 20% | 17.8 ms | 26.2 ms | 0.68× |
| 100,000 | 0.2% | 87.6 ms | 2.5 ms | **35.3×** |
| 100,000 | 1% | 87.7 ms | 7.6 ms | **11.5×** |
| 100,000 | 10% | 82.1 ms | 59.5 ms | **1.4×** |
| 100,000 | 20% | 87.5 ms | 114.1 ms | 0.77× |

So: sparse below roughly 10% density, dense above it, and the sparser and taller
the system the bigger the win. These are Apple MPS numbers; CUDA is unmeasured.

### How the sparse backend avoids reading every constraint

Changing variable `j` perturbs only the constraints in its column. The obstacle is
that the objective is a *global* maximum, so a candidate's score needs the largest
violation among the constraints it did **not** touch.

Keeping the `K` worst violations (`T`) removes it. Everything outside `T` is at
most everything inside it, so if the largest untouched constraint were outside
`T`, it would tie an untouched member of `T`:

> `max(untouched) == max(T \ touched)` whenever `T \ touched` is non-empty.

Carrying `tau`, the `(K+1)`-th violation, closes the remaining case:
`max(max(T \ touched), tau)` is always an upper bound — so accepting on it is
sound — and is exact whenever `T \ touched` is non-empty. A column narrower than
`K` forces that, so the default `K` sits above the widest column and the inexact
case is unreachable. It is counted anyway, in `result.counters`.

## Batches

`solve_batch` runs many instances sharing one `A` in a single tensor program.
Instances are independent; batching helps because one small instance cannot fill a
device on its own. For a *single* instance the parallelism comes from candidates ×
constraints, so one large instance uses the device just as well.

```python
results = minviol.solve_batch(A, lowers, uppers, domain=domain)  # (instances, constraints)
```

## Starting points

| `init` | Meaning |
|---|---|
| `"lstsq_round"` | Least-squares fit to the middle of the finite bounds, rounded onto the domain. The default. |
| `"zero"` | Every variable at the level nearest zero. |
| `"given"` | Use `x0`. |

The cold start is genuinely harder, which is why the single-variable descent
exists. A swap exchanges the levels of two variables, so it preserves the multiset
of assigned levels; from an all-one-level start, a swap-only descent cannot take a
single step (measured: 28.46 unchanged, versus 28.46 → 16.96 with single-variable
moves).

## Install and test

```bash
pip install -e .
pytest
```

Set `MINVIOL_TEST_DEVICE=cuda` or `=mps` to run the suite on a device.
