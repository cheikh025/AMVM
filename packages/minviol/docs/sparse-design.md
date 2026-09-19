# The sparse backend

How it avoids reading every constraint, why that is exact, and what it costs.
Everything here is about one question: a candidate move perturbs only the
constraints in its column, but the objective is a *global* maximum, so how do you
score a candidate without looking at the constraints it left alone?

## The obstacle

For a dense matrix every candidate touches every constraint, so there is nothing
to save and the kernels are large regular blocks. For a sparse matrix, moving
variable `j` changes `A x` only on `supp(col j)`. Per-candidate work should drop
from `m` to `nnz(col j)` — at 100,000 constraints and 0.2% density that is 200
instead of 100,000.

But a candidate's new objective is

```
max( max over UNTOUCHED constraints ,  max over TOUCHED constraints after the move )
```

The second term is cheap. The first is a reduction over `m − nnz` constraints,
which is the whole-matrix pass the saving was supposed to avoid.

## The claim that removes it

Let `v` be the violation vector, `T` the indices of its `K` largest entries, `S`
the touched set and `U` the untouched set.

> **Claim.** `max(U) = max(T \ S)` whenever `T \ S` is non-empty.
>
> *Proof.* Let `i*` maximize `v` over `U`. If `i* ∈ T` then `i* ∈ T \ S` and we
> are done. If `i* ∉ T` then `v[i*] ≤ v[t]` for every `t ∈ T`, by the definition
> of the `K` largest. Pick any `t ∈ T \ S`; since `t ∈ U` and `i*` maximizes over
> `U`, also `v[i*] ≥ v[t]`. The two are equal. ∎

The only property used is that everything outside `T` is at most everything
inside it, which `torch.topk` guarantees. Ties do not break it.

`tests/test_lemma.py` brute-forces this over ~1,600 (system, variable) cases.

### Closing the empty case without a branch

Keep `K+1` entries and let `tau` be the `(K+1)`-th violation. Then

```
Mhat(j) = max( max over T \ S ,  tau )
```

is **always** an upper bound on `max(U)`, and **exact** whenever `T \ S` is
non-empty — because `max(T \ S) >= min(T) >= tau`. Since the candidate's score is
`max(Mhat, touched max)`, an upper bound, accepting a candidate that beats the
incumbent is always sound. What is lost is completeness, and only for candidates
whose true score lies in `[max(T \ S), tau]`.

### And the case need never arise

By pigeonhole, `nnz(col j) < K` forces `T \ S` to be non-empty. So choosing

```
K = max(128, max_j nnz(col j) + 1)
```

makes the fallback unreachable. That is the shipped default, which is why `K`
(`Options.n_filters`) is **not** a tuning knob in the sparse backend — it is
correctness-load-bearing. The inexact count is reported in
`result.counters["untouched_max_inexact"]` anyway, because a bound that is
assumed rather than measured is a bug waiting to be found by someone else.

`tests/test_sparse.py` forces the branch with `K=2` and a column covering every
screened constraint, and asserts the fallback over-estimates rather than
under-estimates. Without that test the branch never executes.

## Three invariants

1. **`K > max_nnz`.** See above.
2. **`T` is keyed on the violation, not on `A x` or on a residual.** With
   one-sided or asymmetric bounds, the largest values of `A x` are not the most
   violated constraints. Keying on the wrong quantity applies the claim to the
   wrong set and silently discards good candidates.
3. **Cache `y = A x`, not `y - b`.** The violation is a function of `y`, and the
   sparse update `y[S] += step * a_S` is its natural form.

## Storage

Not `torch.sparse_csr` or `_csc`: **neither is implemented on the MPS backend**
(`NotImplementedError: new_compressed_tensor`), so a backend built on them would
be CUDA and CPU only. The structure is carried as ordinary tensors:

| Held | Shape | Used for |
|---|---|---|
| `key` (`col*m + row`, sorted) | `nnz` | membership, by binary search |
| `row_idx`, `values` | `nnz` | the column gather |
| `col_ptr` | `n+1` | segment boundaries |
| COO tensor | — | `A @ x`, the one product MPS implements |

Membership — "does candidate `j` touch screened constraint `t`?" — is
`searchsorted` into `key`. Two vectorized ops, however ragged the columns are,
and no dense `(K, n)` row slab to materialize.

The column gather is the standard segmented expansion: `counts` from `col_ptr`,
`repeat_interleave` for the owner, `cumsum` for the offsets, one index into
`row_idx`.

## Swaps touch a constraint twice

A swap's two columns can overlap. Scoring the halves separately would evaluate
one constraint as two independent moves. The entries are therefore merged per
`(candidate, constraint)` — `torch.unique` on `candidate*m + row`, then
`scatter_add` — *before* the violation is evaluated.

The step multiplies the assembled column difference, not each half:

```python
y_new = y_old + delta[candidate] * accumulated      # matches dense exactly
# NOT  y_old + delta*a_left - delta*a_right         # a unit in the last place away
```

`d*(a - b)` and `d*a - d*b` are algebraically equal and not bitwise equal. Since
the test suite compares *trajectories*, that difference is a different program.
With the grouping above, sparse and dense candidate scores agree to float32
resolution and pick the same moves.

They are **not** bitwise equal across platforms, and an earlier version of this
document claimed they were. The two backends reach the same quantity by different
routes -- a dense row block against a gather and a segment reduction -- and a
compiler may contract those differently. On Apple Silicon they agreed bitwise; on
x86-64 they do not. The grouping above is still worth keeping, because it removes
the difference that is *algebraic* rather than incidental, and because the
incremental update has to be grouped the way the candidate was scored or the
realized objective drifts from the predicted one.

## L2 needs none of this

Sums decompose, so the sum of squares is corrected only where the point changed:

```
l2_new = l2_old + sum over S of (v_new^2 - v_old^2)
```

Exact, and proportional to `nnz`.

The square is not what makes this work — additivity over constraints is. The
tie-break is `sum_i (v_i/s)^p` for a fixed `p` and a scale `s` that is constant
for the solve (`Options.tiebreak_power`), and the same correction holds verbatim
with `^p` in place of `^2`. What a tie-break may *not* be is a sum over a set that
moves with the point: that breaks the incremental form here and, more seriously,
stops the quantity being a potential function, so a plateau walk need not
terminate. Experiment 7 in `experiments/algo_memory.md` measured exactly that.

## No screening stage

Screening exists to keep candidates away from a full pass over every constraint.
Here a candidate never costs that, so screening against `K` constraints would
cost more than the answer it protects. The sparse backend's `screen` keeps
everything.

Staged pruning (`prune_first_stage`, `prune_stage_growth`) is likewise dense-only:
splitting a ragged 200-element segment costs more than evaluating it.

## What it costs, measured

256 variables, MPS, backends alternated within each repetition
(`benchmarks/density_crossover.py`):

| Constraints | Density | Dense | Sparse | Speedup |
|---:|---:|---:|---:|---:|
| 20,000 | 0.2% | 18.1 ms | 3.4 ms | 5.3x |
| 20,000 | 5% | 18.0 ms | 7.2 ms | 2.5x |
| 20,000 | 20% | 17.8 ms | 26.2 ms | 0.68x |
| 100,000 | 0.2% | 87.6 ms | 2.5 ms | 35.3x |
| 100,000 | 1% | 87.7 ms | 7.6 ms | 11.5x |
| 100,000 | 10% | 82.1 ms | 59.5 ms | 1.4x |
| 100,000 | 20% | 87.5 ms | 114.1 ms | 0.77x |

Crossover is around 10-15% density, and the win grows with the constraint count.
Sparse is also a bad bet for heavy-tailed columns, where one wide column
serializes a segment the rest of the batch waits on.

### How little of that survives a whole pass

The table above times candidate scoring in isolation. A pass also screens,
refreshes caches, recomputes the top-K and applies a move, and those are `O(m)`
whatever the backend. Measured per pass at 100,000 constraints and 256 variables:

| Density | Single-variable pass | Swap pass | Three full passes |
|---:|---:|---:|---:|
| 0.2% | 4.3x | 0.65x | 0.89x |
| 1% | 1.8x | 0.22x | 0.56x |
| 5% | 1.4x | 0.09x | 0.31x |

**The single-variable pass is a real win; the swap pass is a loss, and it is large
enough to sink the total.** Profiling one swap pass at 0.2% density, 4,130
candidates and K=245: `_untouched_max` 3.3 ms, `_touched` 8.6 ms, of which
`_column_difference` is 5.1 ms and a bare `_gather_columns` is 1.2 ms. The actual
arithmetic is 4,130 x 200 = 826k entries. At roughly a millisecond per operation on
work that small, the pass is bound by kernel launches, not by arithmetic -- so
reading fewer constraints cannot help it.

Two things were tried and did not fix it. Replacing the `torch.unique` merge with
two binary searches (the `_column_difference` above) removed a sort over millions
of keys and made the swap pass slightly *worse*, which is what says the sort was
not the cost. Making `apply_delta` touch only the support, instead of densifying a
column per accepted move, changed the total by less than the noise.

What would fix it is fewer, larger kernels: one candidate list across all level
pairs rather than one per pair, and the two gathers and two lookups of
`_column_difference` fused. Until then, `Options.swap_moves=False` is the honest
recommendation for a sparse problem -- the single-variable neighbourhood is the
natural one for a general constraint system anyway, and swaps are a specialized
2-opt that suits quantization.

## Not done

- **Incremental top-`K`.** `T` is recomputed each pass with an `O(m)` `topk`. At
  100,000 constraints that is the pass boundary, and once per-candidate work is
  proportional to `nnz` it can dominate. Merging `T` with `S` after each move and
  re-taking the top `K` would preserve the invariant at a fraction of the cost.
  Not correctness: `swap_pass` applies at most one move per instance per pass, so
  the pass-start set is never stale.
- **Candidate generation from the active constraints.** A candidate can only
  improve if its column touches a currently-worst constraint — otherwise that
  constraint is untouched and the objective cannot fall. So the improving
  candidates are exactly `supp(row k*)`, which is *row-major* information. This
  would shrink the candidate list as well as the per-candidate work, and it is
  the larger of the two wins. `tests/test_lemma.py` already proves the claim; the
  backend does not yet exploit it.
