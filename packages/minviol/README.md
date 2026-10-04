# minviol

Find a point that satisfies a large system of constraints, on a GPU.

```bash
pip install minviol
```

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

`minviol` minimizes the largest constraint violation over a discrete domain. It is
built for systems with far more constraints than variables, where you want a
workable point quickly rather than a proof of optimality.

## The problem

```
minimize over x, each x_j drawn from a finite sorted domain

    max_k  violation_k(x),   violation_k = max(0, lower_k - (Ax)_k, (Ax)_k - upper_k)
```

An objective of zero means `x` satisfies every constraint. One form covers every
linear row shape:

| Constraint | How to write it |
|---|---|
| `a·x ≤ u` | `lower=None`, `upper=u` |
| `a·x ≥ l` | `lower=l`, `upper=None` |
| `a·x = b` | `lower=upper=b` |
| `l ≤ a·x ≤ u` | both finite |

You do not add slack variables for the inequalities. For `a·x ≤ u` the best slack
is `max(0, u − a·x)`, so it has a closed form and the solver applies it directly.

## When to reach for it

Good fit: many constraints relative to variables, variables discrete over a small
set of values, and a good-enough point beats a certificate.

Reach for an LP/MIP solver instead if you need a proof of infeasibility, an
optimality bound, or continuous variables. `minviol` is a heuristic — a non-zero
result means it did not find a point, not that none exists.

## API

### `solve(A, lower=None, upper=None, *, domain, ...) -> Result`

| Argument | Meaning |
|---|---|
| `A` | `(constraints, variables)` dense tensor, or a `torch.sparse` tensor |
| `lower`, `upper` | `(constraints,)` each; `None` or `±inf` for one-sided rows |
| `domain` | `(levels,)` sorted values the variables may take |
| `budget` | `Budget(seconds=..., stop_when_feasible=True)` |
| `init` | `"lstsq_round"` (default), `"zero"`, or `"given"` with `x0` |
| `fixed` | `(variables,)` bool — variables that must not move |
| `row_scale` | `(constraints,)` — scales each row's violation before the max |
| `options` | `Options(...)`, see below |

Returns a `Result`:

| Field | Meaning |
|---|---|
| `x` | the point, as physical values from the domain |
| `x_index` | the same point as indices into the domain |
| `feasible` | `max_violation <= budget.feasibility_tol` |
| `max_violation` | largest violation, unscaled |
| `objective` | the scaled quantity the search minimized |
| `wall_time`, `iterations`, `counters` | what the solve cost |

### `solve_batch(...) -> list[Result]`

Many instances sharing one `A`, in a single tensor program. Instances are
independent; batching helps because one small instance cannot fill a device on its
own. Shapes gain a leading axis: `lower`, `upper` and `row_scale` become
`(instances, constraints)`, `domain` becomes `(instances, levels)`.

```python
results = minviol.solve_batch(A, lowers, uppers, domain=domain)
```

For a *single* instance the parallelism already comes from candidates ×
constraints, so one large instance uses the device just as well.

### `Options`

The defaults suit general constraint systems and you rarely need to change them.
The ones worth knowing:

| Option | Default | Use it when |
|---|---|---|
| `tiebreak_power` | `2.0` | Set to `4` on integer or otherwise structured systems, where many candidates leave the maximum in exactly the same place. |
| `swap_moves` | `False` | Turn on for problems that start close to the answer with roughly the right distribution of values. |
| `seed` | `9101` | Reproducibility; the solver is deterministic given a seed and a budget. |
| `memory_budget_mb` | `256` | Bound the largest intermediate tensor. |

## Dense or sparse

`solve` follows the type you hand it: a dense tensor uses the dense backend, a
`torch.sparse` tensor uses the sparse one. It does not convert behind your back,
because silently densifying a matrix you deliberately kept sparse is a surprising
way to run out of memory.

Sparse pays below roughly 10% density and the gain grows with the constraint
count. Candidate scoring, 256 variables:

| Constraints | Density | Dense | Sparse |
|---:|---:|---:|---:|
| 100,000 | 0.2% | 87.6 ms | **2.5 ms** |
| 100,000 | 1% | 87.7 ms | **7.6 ms** |
| 100,000 | 10% | 82.1 ms | **59.5 ms** |
| 100,000 | 20% | 87.5 ms | 114.1 ms |

Measured on Apple Silicon (MPS); CUDA is not yet benchmarked. Reproduce with
`benchmarks/density_crossover.py`.

Dense memory is `O(constraints × variables)` — 262,144 × 768 in float32 is about
805 MB. Sparse is `O(nnz)`.

## Starting points

| `init` | What it does |
|---|---|
| `"lstsq_round"` | Least-squares fit to the constraints' targets, rounded onto the domain. The default, and usually much better than starting cold. |
| `"zero"` | Every variable at the level nearest zero. |
| `"given"` | Use the `x0` you pass. |

## Feasibility tolerance

`Budget.feasibility_tol` defaults to `None`, meaning it is derived from the dtype,
the bound magnitudes and the constraint count. Computing `A x` over many terms
accumulates rounding, so on 100,000 float32 constraints with bounds of order 100
no point can be shown to satisfy them below about `4e-3`. A fixed small threshold
would make the solver find an answer and then refuse to report it.

Pass a number to say what your problem counts as satisfied.

## Scope

- Variables are discrete: a finite sorted domain, shared by all variables of an
  instance. Continuous variables are not supported.
- The domain should be small — a pass costs one kernel launch per level pair.
- `A` is a plain matrix of coefficients; there is no modelling layer.

## Examples

```bash
python examples/feasibility.py     # integer system with a planted solution
python examples/tomography.py      # image reconstruction from projections
```

## Development

```bash
pip install -e ".[test]"
pytest
```

Set `MINVIOL_TEST_DEVICE=cuda` or `=mps` to run the suite on a device. Design
notes are in `docs/`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
