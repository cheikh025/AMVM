# Where the remaining gap is, and what to try next

Written after the acceleration work merged. Everything here rests on two
measurements that had not been taken before: what a solved row actually looks
like, and how much the objective is still moving after the production budget
expires. Both changed the ranking, and both argue against spending more effort on
raw speed.

Measured on Apple Silicon MPS, real opt-125m layers with wikitext2 activations at
N=16,384, 3 bits, `linf_l2_nonincrease`. Reproduce with
`experiments/audit_solution_structure.py`.

## What the audit found

**The search has not converged when the production budget ends.** One row, one
seed, objective reached by elapsed time:

| Layer | initial | 10 s | 30 s | 60 s | 300 s | gain, 10 s to 300 s |
|---|---:|---:|---:|---:|---:|---:|
| fc1 | 0.09268 | 0.06681 | 0.06653 | 0.06653 | 0.06459 | 3.3% |
| q_proj | 0.82622 | 0.57947 | 0.57189 | 0.55943 | 0.53495 | 7.7% |
| fc2 | 0.53130 | 0.04481 | 0.04228 | 0.03978 | 0.03723 | 16.9% |

**Run-to-run variance is larger than everything the acceleration work bought.**
Best against worst of three seeds on the same row at the same 10 second budget:

| Layer | median spread across rows | worst row |
|---|---:|---:|
| q_proj | 2.2% | 4.0% |
| fc1 | 4.0% | 5.3% |
| fc2 | 16.2% | 26.5% |

The whole Tier A stack moved the median objective 1.3% to 1.6%. Seed choice is
worth more than that on every layer, and ten times more on `fc2`.

**The active set is tiny.** Structure of the returned solutions:

| Property | fc1 | fc2 | q_proj |
|---|---:|---:|---:|
| Samples attaining the maximum | 1 | 1 | 1 |
| Samples within 1% of it | 7 | 10 | 3 |
| Samples within 10% of it | 67 | 76 | 61 |
| Variables changed from round-to-nearest | 6.1% | 4.4% | 11.5% |
| Variables moved more than one level | 1 | 1 | 0 |
| Share of total L2 held by the top 100 samples | 6.8% | 8.8% | 7.1% |

Three consequences. The objective is decided by one sample out of 16,384, while
every candidate is scored against all of them. The optimum sits within one level
of round-to-nearest for essentially every variable, so widening the level
neighborhood has little room. And the sum-of-squares gate constrains a quantity
that the top 100 samples barely participate in, while the infinity norm is
decided inside that top 100, so the two criteria pull on different parts of the
residual.

### Gap decomposition

| Source | Measured ceiling | Where the number comes from |
|---|---|---|
| Under-search on `fc2` | 16.9% (time), 16.2% (seed) | table 1, table 2 |
| Under-search on `q_proj` | 7.7% (time), 2.2% (seed) | table 1, table 2 |
| Under-search on `fc1` | 3.3% (time), 4.0% (seed) | table 1, table 2 |
| Raw speed, already harvested | 1.3% to 1.6% per 2x | Tier A equal-time result |
| Level-neighborhood width | small | solutions move one level |
| Acceptance policy | unmeasured | never compared on real instances |

Anything that only makes the current search faster is bounded by the fourth row.
The first three rows are five to ten times larger, and both routes into them run
through the same place: the inner loop scores candidates against all N samples to
compute a maximum that one sample decides.

### The scale wall

Production is CUDA at N=262,144 with several processes per GPU, and the README
already reports out-of-memory on 16 GB cards for `fc1`.

| Quantity | N=16,384 | N=262,144 |
|---|---:|---:|
| Activation matrix, `fc2` | 0.20 GB | 3.22 GB |
| One candidate tile, 4,096 candidates by N | 0.27 GB | 4.29 GB |
| Working set of 256 samples by 4,096 candidates | 4 MB | 4 MB |

The candidate tile is the binding term and it grows with N. A working set does
not. That is the same conclusion the gap decomposition reaches, from a different
direction.

## Tier 1, worth doing now

### 1. Evaluate candidates on a working set, verify the winner

**Gap attacked:** under-search, the largest measured ceiling.

The classical method for Chebyshev approximation problems does not evaluate every
constraint: it keeps an active set, solves against it, checks the solution
against the full constraint set, and exchanges in whatever was violated. That is
the Remez exchange, and in linear programming terms it is row generation. This
solver instead evaluates every candidate against all 16,384 samples to find a
maximum that one sample determines.

The change is small because the machinery already exists. The full residual is
already maintained incrementally, and the top-`num_partial` samples are already
extracted for `L_set` every time the residual is refreshed, so the working set is
free. What changes is that the exact stage scores candidates on the working set
instead of all samples, and only the selected candidate is verified against all
of them. If verification finds a larger true maximum, the violating sample joins
the working set and selection repeats. Growth of the working set guarantees
termination, and verification makes the accepted move exactly as sound as today.

Sizing, on the measured profile: the exact stage was 962M row-candidate products
per 8 iterations on `q_proj`, against 2.1M for screening. A 256-sample working set
cuts the dominant term by 64x here and by 1,024x at production N, plus one
full-N verification per accepted move, of which there were 11 in those 8
iterations. Even a fraction of that converts into iterations, and the 10 s to
300 s curve says iterations are worth 3% to 17%.

The honest risk is that the working set misses the sample that ends up binding,
making verification fail often and the exchange loop thrash. The audit says that
is unlikely, since 3 to 10 samples sit within 1% of the maximum and the working
set is the top 100 by residual, but the verification failure rate is the first
thing to measure and it decides the idea.

**Device note.** This is friendly to both backends: it shrinks tensors rather
than adding control flow, the verification is one matrix-vector product, and it
needs no float64 and no custom kernel. It is the only proposal here that
*removes* the term that blocks production scale.

```python
# src/GPU/ALNS/candidate_scores.py
from typing import NamedTuple

class Scores(NamedTuple):
    maxima: torch.Tensor      # (candidates,) infinity norm, exact or a lower bound
    squares: torch.Tensor     # (candidates,) sum of squares over the evaluated samples
    exact: bool               # whether these were computed over every sample

def candidate_scores(state, left, right, physical_delta, *,
                     method: str = "working_set_verified",
                     working_set: torch.Tensor | None = None,
                     prune_bound: torch.Tensor | None = None) -> Scores:
    """method: "all_samples" | "working_set" | "working_set_verified"."""
```

Tests to write first: `working_set_verified` picks the same swap as
`all_samples` on small instances across seeds and all three acceptance policies;
a constructed instance whose binding sample is outside the working set is caught
by verification and not applied; the exchange loop terminates and the working set
grows monotonically; the residual after an applied move still agrees with direct
recomputation; the verification failure rate is reported as a counter.

### 2. Measure the acceptance policy on real instances

**Gap attacked:** under-search, through the number of moves the search can make.

`linf_l2_nonincrease` is the default for quantization and has never been compared
against the alternatives on real data. Two measurements suggest it is the binding
constraint on movement. Greedy repair applied zero moves in 12 of 12 selections,
so roughly half the roulette wheel's picks do nothing at all. And the sum of
squares that the gate protects is a quantity the top 100 samples hold only 7% to
9% of, so the gate is mostly refusing moves for disturbing the bulk of the
residual while the objective is decided elsewhere.

This is a one-flag experiment on an existing harness, costs an afternoon of
machine time, and has three possible outcomes, all informative. It should run
before anything else on this list, because a policy change would alter what every
other idea is measured against.

```bash
for policy in linf linf_l2_tiebreak linf_l2_nonincrease; do
  experiments/run_protocol.sh policy-$policy quality   # with --acceptance-policy $policy
done
python experiments/summarize_results.py policy-linf policy-linf_l2_tiebreak policy-linf_l2_nonincrease
```

### 3. Spend the budget on replicas instead of one long run

**Gap attacked:** path dependence, worth 2.2% to 16.2%.

The seed spread means one run is a draw from a wide distribution, and on `fc2` the
draw matters more than anything else measured. Running several replicas of the
same row with different seeds and keeping the best converts that variance from a
liability into a resource. The batched module already holds independent problems
over a shared activation matrix, so replicas of one row are the same code with
the same weights and different seeds.

This is not free. Batching reached parity rather than speedup, so R replicas cost
about R times the work, and each gets a fraction of the budget. Whether best of
three at 10 s beats one run at 30 s is genuinely open: the seed spread on `fc2` is
16.2% while 10 s to 30 s buys 5.6%, which favours replicas, but on `fc1` the
spread is 4.0% and the same time step buys almost nothing, which does not.
Measure per layer, and expect the answer to differ by layer.

## Tier 2, and what is not worth pursuing

**Tier 2, after the above.**

- **Apply several non-conflicting swaps per pass.** A pass costs 50 to 1,600 ms
  and applies exactly one move out of thousands that clear the infinity-norm
  bound. Swaps on disjoint variables could be applied together with one combined
  verification and a rollback. Missing number first: how many candidates clear
  the sum-of-squares gate, not just the infinity-norm bound. Idea 2 may remove
  the gate and change this entirely.
- **Make the budget mean something on `fc2`.** One iteration overruns the 10 s
  budget by 18% here and would exceed it outright at production N, because the
  stopping rule is checked only between iterations. The batched path already
  checks between passes. Worth doing regardless, since it makes every equal-time
  comparison cleaner.
- **Switch the batched path to incumbent pruning.** It still prunes against the
  acceptance bound, which was measured as removing 0.15% of the work, and pays a
  compaction synchronization per stage for it.

**Not pursued, with reasons.**

- **Wider level neighborhoods.** Solutions move one level, with a single variable
  moving two across all three layers. Little room.
- **Gram-matrix L2.** The term it removes is 10% of a stage that idea 1 shrinks
  by 64x, so its share of the total falls further.
- **Transposed activations.** Gathers are not the bottleneck on this device, a
  second copy of `fc2` is 3.2 GB at production scale, and idea 1 shrinks the
  gathered tensor by the same 64x.
- **Fused custom kernels.** Not available on MPS without writing Metal, and idea 1
  removes most of the work a kernel would fuse. Revisit only if a CUDA profile
  shows the working-set loop dominating.
- **Exact solve for a bound.** Tempting for measuring the true optimality gap, but
  it does not work here. A subset-of-samples relaxation is trivially zero, because
  a few hundred constraints against 768 integer variables are hugely
  underdetermined, and the full problem is 6,144 to 24,576 integer variables. The
  long-run curve in table 1 is the practical substitute.

## Sequence

Idea 2 first, because it is cheap and it changes the baseline every later
measurement is taken against. Then idea 1, evaluator before search, with the
verification failure rate as the go or no-go signal. Then idea 3, per layer, on
whatever policy idea 2 selected. Tier 2 after.

All of this is MPS at one sixteenth of production sample count. The eight checks
in `docs/gpu-acceleration.md` still gate whether any of it transfers, and idea 1
is the one whose value grows rather than shrinks with the move to production N.
