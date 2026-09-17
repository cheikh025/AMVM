# Algorithm Experiment Memory

## Project Profile

**Project name:** AMVM
**Problem class:** adaptive large-neighborhood search for discrete matrix-vector minimax approximation

### Tracked metrics

| Metric | Direction | Noise floor | Notes |
|---|---|---|---|
| Exact residual agreement | must pass | — | cached residual and objective versus direct `Ax-b` |
| Feasible-domain membership | must pass | — | includes fixed variables and nonuniform domains |
| Infinity norm | lower is better | unknown | compare under equal time and seeds |
| Peak swap memory | lower is better | — | analytical tensor bound plus CUDA peak allocation when available |
| Runtime | lower is better | measure per device | runtime is hardware-dependent |

### Hard constraints

1. Every accepted move must agree with direct residual recomputation.
2. Returned values must belong to the caller-provided discrete domain.
3. Fixed variables must remain unchanged in all operators.
4. Unexpected runtime errors must propagate; fallbacks must be explicit and recorded.
5. Algorithm-quality comparisons must use equal time budgets and multiple owned seeds.

### Evaluation harness

- **Correctness command:** `/tmp/amvm-validation/bin/python -m pytest -q`
- **Swap benchmark:** `/tmp/amvm-validation/bin/python experiments/benchmark_swap_tiles.py --variable-tile 256 --row-tile 256 --candidate-chunk 512 --output experiments/results/swap_tiles_cpu_default.json`
- **Benchmark instance set:** tiny deterministic unit/integration cases; no full model-quality benchmark is checked into this repository
- **Result format:** pytest output and JSON in `experiments/results/`
- **Solver extension point:** `src/GPU/ALNS/`

## Baseline (as of session start)

| Metric | Value |
|---|---|
| Regression suite | 11 passed, 11 expected failures; tomography import failed until its declared dependency was installed |
| Full quality benchmark | unavailable |
| CUDA peak memory | unavailable on this host |

**Baseline solver:** commit `68501f61302505f6b124df1516945c40bd935d86`
**Last baseline run:** 2026-09-15
**Hardware:** Apple arm64 macOS host, PyTorch 2.13.0, CPU only

## Second baseline (entries 3 and later)

| Item | Value |
|---|---|
| Baseline commit | `5dc30c6`, measured in a pristine worktree |
| Hardware | Apple Silicon, MPS backend, PyTorch 2.13 |
| Instances | opt-125m layer 0 `q_proj` and `fc1` (M=768), `fc2` (M=3072), real wikitext2 activations, N=16,384 |
| Protocol | `experiments/run_protocol.sh <label>`; speed is 20 iterations over 4 rows, quality is a 10 second budget over 4 rows and 3 seeds |
| Decision metric | median paired change in the final infinity norm at equal time, with no pair allowed to get worse |

Production runs CUDA at N=262,144 with several processes per GPU. None of the
entries below were measured there; `docs/gpu-acceleration.md` lists what that
needs.

**Baseline speed (median ms per iteration):** q_proj 288, fc1 213, fc2 2850
**Baseline quality (median final infinity norm):** q_proj 0.6241, fc1 0.0616, fc2 0.0443

## Measurement correction, 2026-09-17

Entries 3 to 9 were measured in blocks: one variant run to completion, then the
next. The blocks spanned several hours during which the machine ran on battery
and the charge fell, and battery and wall power use different performance modes
here. Re-measured by alternating the builds within each repetition, the
end-to-end stack of entries 3 to 6 is 1.87x (q_proj), 2.03x (fc1) and 1.62x
(fc2), against 1.99x, 2.03x and 1.76x reported. Read the headline as 1.6x to 2.0x.

Power state itself was not the cause: the same configuration on battery and on
wall power differs by under 7%, in both directions, and low power mode was off.
The block design simply could not cancel drift of any kind, and the drift fell
in the flattering direction on two layers out of three.

The equal-time quality results are not corrected and their direction is safe:
variants ran in ladder order while the battery drained, so drift handicapped the
later variants, which are the ones that won.

Unaffected, because they do not depend on timing: the trajectory equivalence
results, the whole test suite, and the structural audit of solved rows.

**Rule going forward:** no timing claim from block measurement. Use
`experiments/run_interleaved.sh`, which measures several variants in one process
with the solves interleaved. This is not extra work: it runs exactly the solves
the old protocol ran, reordered, in fewer processes. `experiments/ab_interleaved.sh`
remains for the one case flags cannot express, comparing different commits.

## Running summary

| # | Date | Title | Layer | Decision | Correctness | Memory | Runtime |
|---|---|---|---|---|---|---|---|
| 9 | 2026-09-16 | Batched multi-row solver (Tier B) | Infrastructure | PARITY, kept behind a flag | feasible, objectives honest | rows x samples residual block | parity at equal time, no speedup on MPS |
| 8 | 2026-09-16 | Larger candidate chunk | Infrastructure | REVERT | exact | larger intermediates | 1.08x-1.17x slower |
| 7 | 2026-09-16 | Early abort against the acceptance bound | Search | REVERT | exact | neutral | removed 0.15% of work |
| 6 | 2026-09-16 | Pruning against the incumbent best | Search | KEEP (flag, default off) | same swap chosen | neutral | 1.21x on fc2, 1.16x slower on q_proj |
| 5 | 2026-09-16 | Budget-derived row tile | Infrastructure | KEEP | identical objective | bounded by an explicit budget | 1.13x-1.16x over entry 4 |
| 4 | 2026-09-16 | Device-resident swap reduction | Infrastructure | KEEP | identical objective | neutral | 1.14x-1.29x over entry 3 |
| 3 | 2026-09-16 | Host-side index gathering and settled-pass skip | Infrastructure/Search | KEEP | exact; skip is statistically equivalent | neutral | 1.17x-1.46x over baseline |
| 2 | 2026-09-15 | Bounded exhaustive swap evaluation | Infrastructure | KEEP | exact match | 4x smaller filter tensor in measured default case | 2.54x slower on CPU microbenchmark |
| 1 | 2026-09-15 | Explicit domains and physical move updates | Evaluator/Search | KEEP | 27/27 tests pass | neutral | full benchmark unavailable |

---

## Experiment 9: Batched multi-row solver (Tier B)

**Date:** 2026-09-16
**Idea layer:** Infrastructure
**Decision:** PARITY, kept behind a flag and not enabled

### Idea card

Every row of a matrix shares the activation matrix and is independent, so hold a
group of rows in one tensor program: `(rows, variables)` weights against
`(samples, rows)` residuals, one candidate list carrying a row tag, one winner
per row from three segment reductions. Rows that stop improving are masked out
of the candidate build. The `alns` package cannot drive this, because it runs one
instance per Python loop and owns selection and acceptance, so the batched path
has its own small loop and the package keeps serving tomography and FIR design.

### Implementation surface

- Files added: `src/GPU/ALNS/batched.py`, `tests/test_batched_rows.py`,
  `experiments/bench_batched_rows.py`
- Files changed: `src/GPU/full_layer.py` (a batched branch writing the same
  per-row files the workers write)
- Feature controls: `AMVM_BATCHED_ROWS`, `AMVM_BATCH_SIZE`,
  `AMVM_BATCH_CANDIDATE_TILE`

### Results

Equal total time: the per-row path gets the production budget per row, the
batched path gets the sum of those budgets for the whole group.

| Instance | Rows | Per-row path | Batched path | Rows better | Median relative change |
|---|---:|---|---|---:|---:|
| q_proj (M=768) | 16 | 161.3 s, 0.654913 | 165.1 s, 0.659705 | 5 of 16 | +2.53% |
| q_proj (M=768) | 64 | 646.5 s, 0.643139 | 640.1 s, 0.654430 | 28 of 64 | +1.40% |
| fc2 (M=3072) | 16 | 189.5 s, 0.043478 | 160.2 s, 0.043536 | 7 of 16 | +1.36% |

Positive change means the batched path ended worse.

### Analysis

Parity, not a win. The premise of Tier B is that a single row does not fill the
device, so rows should be held together. On this device that premise no longer
holds once Tier A is in: the per-row program already keeps the device busy, and
what batching adds is amortized Python and launch cost that Tier A had already
removed.

Batch size moves the result in the expected direction without crossing over.
Going from 16 to 64 rows raised the share of rows the batched path wins from 31%
to 44%. Rows are also lock-stepped, so each row gets the same number of
perturbation rounds instead of its own budget, which costs the rows that would
have used more.

One unrelated observation worth carrying: the per-row path overran its budget by
18% on fc2, because the stopping rule is only checked between iterations and an
fc2 iteration is long. The batched path now checks between passes and lands on
its budget.

Kept behind `AMVM_BATCHED_ROWS`, not enabled. The comparison that would justify
enabling it is against ten worker processes per GPU on CUDA, which is what
production actually runs, not against one sequential loop.

### Lessons

- An idea premised on an idle device has to be re-argued once the device is no
  longer idle. Tier A removed the very overhead that made Tier B attractive.
- The batched path still prunes against the acceptance bound, the strategy
  entry 7 measured at 0.15% of work removed, and pays a compaction
  synchronization per stage for it. On `fc2` that is roughly twenty stages of
  overhead per level pair buying nothing, so the parity result understates the
  path slightly. Switching it to prune against each row's incumbent, as the
  single-row path does, is the obvious next change if it is ever revisited.
- The group budget is `seconds * group size`, which matches a sequential device
  and not the production fan-out where rows run concurrently. Anyone enabling the
  flag on CUDA has to revisit the budget first.
- Two defects survived a first round of tests that looked thorough and were
  caught only by benchmarking. Reading level values positionally rather than by
  row index is invisible while every row shares one domain and every row
  improves. Treating a row with no admissible candidate as a row whose candidates
  all tie is invisible while the screen is tight enough that every row has one.
- A test that checks only the final answer hides both, because the driver
  recomputes residuals from scratch before returning. Assert on the cached state
  during the search, not only on what comes out.

### Artefacts

- Raw results: `experiments/results/batched/`
- Relevant code: `src/GPU/ALNS/batched.py`

---

## Experiment 8: Larger candidate chunk

**Date:** 2026-09-16
**Idea layer:** Infrastructure
**Decision:** REVERT

### Idea card

Evaluate more candidates per chunk so the incumbent bound propagates sooner and
pruning starts earlier.

### Results

| Configuration | q_proj | fc2 |
|---|---:|---:|
| chunk 512, no pruning | 150 | 1619 |
| chunk 32768, no pruning | 163 | 1941 |
| chunk 512, pruning | 189 | 1350 |
| chunk 32768, pruning | 151 | 2238 |

Median milliseconds per iteration, from
`experiments/results/chunk-{512,32768}-{no,}pruning/`.

### Analysis

A regression in every combination. The larger intermediates cost more than the
earlier bound saves. The profiler-on sweep that first suggested this was
misleading: profiler overhead scales with operation count, which is exactly what
the chunk size changes, so it flattered the large-chunk configurations.

### Lessons

- Never choose a tile size from a profiler-on measurement.

---

## Experiment 7: Early abort against the acceptance bound

**Date:** 2026-09-16
**Idea layer:** Search
**Decision:** REVERT

### Idea card

Visit samples worst residual first and drop a candidate as soon as its running
maximum passes the acceptance bound, so hopeless candidates stop costing samples.

### Results

| Metric | Value |
|---|---:|
| Row-candidate products, unpruned | 962.2M |
| Row-candidate products, with the abort | 960.8M |
| Work removed | 0.15% |
| Screening survivors rejected by any sample | 0 of 2,000 |

### Analysis

The idea rests on an assumption that measurement refuted: that many screening
survivors fail the acceptance bound somewhere. They do not. Screening already
uses the samples with the largest residuals, so a candidate that passes it stays
under the bound essentially everywhere, and there is nothing left to abort. Of a
sample of 2,000 survivors, not one was rejected by any sample. Ordering samples
by residual magnitude, by row norm, by largest entry, or by the spread of the two
swapped columns made no difference, because none of them predicts a rejection
that does not happen.

What the same measurement showed is that the useful bound is the best candidate
found so far, not the acceptance bound, which became experiment 6.

### Lessons

- Measure the assumption behind a pruning rule before implementing the rule. The
  measurement cost twenty minutes; the implementation cost two hours.

### Artefacts

- Diagnostics: `experiments/diagnose_kill_order.py` (which sample ordering rejects
  survivors fastest, answer: none of them) and
  `experiments/diagnose_prune_potential.py` (work remaining when pruning against
  the true best). Both print to standard output rather than writing a result
  file.

---

## Experiment 6: Pruning against the incumbent best

**Date:** 2026-09-16
**Idea layer:** Search
**Decision:** KEEP behind a flag, default off

### Idea card

Selection keeps the candidate with the smallest infinity norm, so a candidate
whose running maximum has already passed the best one found so far can never be
chosen and its remaining samples are dead work. The comparison is strict, so
candidates that can still tie survive and stay able to win on the sum of squares.
Sample stages grow as the live set shrinks, holding the intermediate near a fixed
size instead of collapsing into many small launches.

### Results

| Metric | q_proj | fc2 |
|---|---:|---:|
| Oracle work remaining, pruning against the true best | 44% | 8.5% |
| Milliseconds per iteration, 20 iterations, without | 150 | 1619 |
| Milliseconds per iteration, 20 iterations, with | 189 | 1350 |
| Iterations reached in a 10 second budget, without | 100 | 3 |
| Iterations reached in a 10 second budget, with | 126 | 6 |
| Paired rows worse at equal time | 0 of 12 | 0 of 12 |

### Analysis

The saving grows with the number of candidates per level pair and with the sample
count, while the cost is one synchronization per stage, so wide layers pay it off
and narrow ones do not. At a fixed 20 iterations it looks like a regression on
the narrow layer, but at a fixed time budget it reaches more iterations on every
layer. The two measures disagree because iteration cost is not stationary: early
iterations do far more local-search work than late ones, so a fixed-iteration
measure over-weights the expensive early ones. The time-budget measure is the one
production experiences.

Kept behind a flag rather than turned on by default because the margin on narrow
layers is within noise on this device, and because the balance is expected to
move at the production sample count, where the saving grows and the cost does
not.

### Lessons

- A fixed-iteration speed measure can invert the ranking a time budget gives.
  Quote both, and decide on the time budget.

### Artefacts

- Raw results: `experiments/results/a6-prune-incumbent/`, the chunk isolation in
  `experiments/results/chunk-*/`, and the iteration-cost drift in
  `experiments/results/drift/`, where the 20-iteration and 100-iteration runs
  reach identical objectives, confirming that pruning changes only the cost.

---

## Experiment 5: Budget-derived row tile

**Date:** 2026-09-16
**Idea layer:** Infrastructure
**Decision:** KEEP

### Idea card

Derive the row tile from a memory budget instead of the fixed 256 rows chosen
for CPU memory, so the tile adapts to the sample count and the device while peak
memory stays explicitly bounded.

### Results

| Metric | q_proj | fc1 | fc2 |
|---|---:|---:|---:|
| Milliseconds per iteration | 145 | 105 | 1619 |
| Gain over entry 4 | 1.26x | 1.16x | 1.17x |
| Operations per iteration | 5,200 | - | - |

Operations per iteration fell from 46,900 to 5,200 on q_proj. The objective is
identical at every iteration count tested.

### Lessons

- Express a tile as a memory budget, not a row count. The same setting then means
  the same thing on a laptop and on a datacenter GPU.

---

## Experiment 4: Device-resident swap reduction

**Date:** 2026-09-16
**Idea layer:** Infrastructure
**Decision:** KEEP

### Idea card

Keep the running best of a swap pass in device tensors updated by a masked
comparison, instead of copying four scalars per candidate chunk to the host and
asking the host which is better. Hoist the domain to the host once so level
comparisons never synchronize, and drop the early-exit emptiness checks.

### Results

| Metric | Before | After |
|---|---:|---:|
| Synchronizations per iteration | 421 | 155 |
| Scalar reads per iteration | 357 | 128 |
| Milliseconds per iteration, q_proj | 207 | 183 |
| Milliseconds per iteration, fc2 | 2440 | 1887 |

Selection is unchanged: same infinity norm, same tie rule, same earliest
candidate on a full tie, verified against the bounded implementation across
seeds, tile shapes and all three acceptance policies.

### Lessons

- The cost of a scalar read is not the copy, it is the wait. Reducing on the
  device and reading once per pass is worth far more than the arithmetic it adds.

---

## Experiment 3: Host-side index gathering and the settled-pass skip

**Date:** 2026-09-16
**Idea layer:** Infrastructure and Search
**Decision:** KEEP

### Idea card

Two independent changes with the same cause, that the solver spent its time
waiting on the host.

Gather removed indices and current levels in one transfer instead of one
synchronization per variable. Separately, skip a local-search pass whose starting
point is already settled, meaning an earlier pass examined it without finding a
move and nothing has moved since. Whether a pass finds a move is a property of
the point and not of the random draws, because the pass evaluates every candidate
and keeps the best, so a repeat pass on an unchanged point finds nothing.

### Results

| Metric | Value |
|---|---:|
| Removed-index gather at M=768, before | 113 ms per call |
| Removed-index gather at M=768, after | 1 ms per call |
| Iterations whose local-search pass was a guaranteed no-op | 12 of 20 |
| Milliseconds per iteration, q_proj | 288 to 207 |
| Milliseconds per iteration, fc2 | 2850 to 2440 |
| Paired rows worse at equal time | 0 of 36 |

### Analysis

The skip is statistically equivalent rather than trajectory identical, because
the skipped pass would have drawn from the random stream. It is therefore judged
at equal time, where no paired row got worse on any layer.

Fixing the index gathering also exposed an inconsistency worth keeping fixed:
`worst_remove` built its removed mask as a numpy array while `random_remove`
built a device tensor, so consumers had to handle two container types.

### Lessons

- Indexing a device tensor inside a Python loop is a synchronization per element.
  It is the cheapest large win in a solver of this shape and the easiest to miss,
  because the line looks like ordinary Python.

---
## Experiment 2: Bounded exhaustive swap evaluation

**Date:** 2026-09-15
**Idea layer:** Infrastructure
**Decision:** KEEP

### Idea card

Tile both variable groups and the residual rows, retain a two-dimensional
survivor mask per variable tile, and evaluate every survivor in bounded chunks.
This removes the unguarded three-dimensional allocation and arbitrary first-100
candidate cutoff while preserving the best exact candidate.

### Implementation surface

- Files changed: `src/GPU/ALNS/local_search.py`, `experiments/benchmark_swap_tiles.py`
- Feature controls: `variable_tile`, `row_tile`, `candidate_chunk`

### Results

| Metric | Dense | Tiled | Delta |
|---|---:|---:|---:|
| Best objective | 10.8149977 | 10.8149977 | exact match |
| Filter tensor bound | 32 MiB | 8 MiB | -75% (4x reduction) |
| Median CPU runtime | 0.0266 s | 0.0676 s | +0.0410 s (2.54x slower) |

For the reported 100 by 2,048 by 2,048 float32 filter, the same 256-variable
tiling bounds that tensor at 25 MiB instead of 1.56 GiB, a 64x reduction.

### Analysis

The memory mechanism is validated and candidate selection is exact on the
synthetic benchmark. CPU runtime regressed because tiling adds Python loop
overhead. This is a memory/correctness KEEP, not a speed claim. CUDA runtime and
peak allocation still require measurement on the target GPU.

### Lessons

- Bound allocations before constructing candidate tensors.
- Preserve all surviving candidates through chunked evaluation.
- Tune tile sizes per device; CPU timing does not predict CUDA timing.

### Artefacts

- Raw results: `experiments/results/swap_tiles_cpu_default.json`
- Relevant code: `src/GPU/ALNS/local_search.py`

---

## Experiment 1: Explicit domains and physical move updates

**Date:** 2026-09-15
**Idea layer:** Evaluator/Search
**Decision:** KEEP

### Idea card

Represent the discrete domain explicitly and calculate every residual update
from physical level differences. Separate read-only move evaluation from move
application, refresh residuals immediately, and enforce one fixed-variable mask.

### Implementation surface

- Files changed: `State.py`, repair/remove/local-search operators, `ALNS.py`, `tomograph.py`
- Feature controls: explicit `discrete_domain` and `acceptance_policy`

### Results

| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| Passing tests | 11 | 27 | +16 |
| Expected known failures | 11 | 0 | -11 |
| Direct residual checks | mixed | all pass | corrected |

### Analysis

The executable counterexamples now agree with direct matrix multiplication,
including sequential repair, nonuniform levels, tomography domains, outliers,
float64 updates, retry selection, and unexpected error propagation. A full
model-quality comparison is not available in this checkout.

### Lessons

- Store domain values at the solver boundary.
- Use assignment-based moves instead of mutable step-size inference.
- Keep optional L2 behavior explicit; pure DMMV defaults to infinity norm.

### Artefacts

- Regression tests: `tests/`
- Plan and evidence scope: `docs/implementation-plan.md`

---
