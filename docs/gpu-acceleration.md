# GPU acceleration of the ALNS row solver

What was changed, what it was measured to do, and what could not be measured
here. Every number below was taken on an Apple Silicon MPS device. Production
runs on CUDA at four times the sample count and with several processes per GPU,
so nothing here transfers to CUDA without being re-measured. The open CUDA
checks are listed at the end with the exact commands.

## Measurement setup

| Item | Value |
|---|---|
| Device | Apple Silicon, MPS backend, PyTorch 2.13 |
| Instances | opt-125m layer 0 `q_proj` (M=768), `fc1` (M=768), `fc2` (M=3072) |
| Activations | wikitext2, 8 sequences of 2048 tokens, giving N=16,384 rows |
| Bits | 3 |
| Acceptance | `linf_l2_nonincrease` |
| Baseline | commit `5dc30c6`, measured in a pristine worktree |

Production uses N=262,144 and runs ten rows per GPU as separate processes. The
instances here are real weights and real activations at one sixteenth of the
production sample count.

Reproduce the inputs with `experiments/capture_real_instances.py`; the files are
not checked in. Run a variant with `experiments/run_protocol.sh <label>`, and
read the tables with `experiments/summarize_results.py`. For a timing claim, use
`experiments/run_interleaved.sh`, which measures several variants in one process
with the solves interleaved instead of blocked, at no extra cost and in the same
output layout; see the correction under "Changes and what each one bought" for why
that matters.

## What the profile found

Three facts drove every change.

- **The solver waited on the host, it did not compute.** About 600 host
  synchronizations per ALNS iteration, most of them scalar reads inside the swap
  pass and one per variable inside the repair operators.
- **Most local-search passes were guaranteed no-ops.** In a 20-iteration run,
  `greedy_repair` was selected 12 times and applied zero moves every time. Each
  of those was followed by a full swap pass over an unchanged point that a
  previous pass had already examined.
- **Screening barely filters, and the exact stage is where the work is.** Of
  2.1 million screened pairs, 2.8% survived, and every survivor was then scored
  against all N samples. Of those survivors only one per pass is ever applied.

A fourth fact killed an idea: pruning candidates against the *acceptance* bound
saves nothing. Nearly every screening survivor stays under that bound on every
sample, because screening already uses the samples with the largest residuals.
Measured directly, zero of 2,000 survivors were rejected by any sample. The
useful bound is the best candidate found so far, not the acceptance bound.

## Changes and what each one bought

Cumulative, each row adding one change to the row above, so the difference
between neighbouring rows is attributable to that change.

| Variant | q_proj | fc1 | fc2 |
|---|---:|---:|---:|
| baseline | 288 | 213 | 2850 |
| A2 host-side index gathering | 223 | 155 | 2527 |
| A1 skip settled local search | 207 | 147 | 2440 |
| A3 device-resident swap pass | 183 | 122 | 1887 |
| A4 budget-derived row tile | 145 | 105 | 1619 |

Median milliseconds per iteration, 20 iterations, four rows, profiler off.

**These were measured in blocks, one variant after another, and the totals they
give are 6% to 8% too high on two of the three layers.** The blocks ran over
several hours while the machine was on battery and the charge fell, and battery
and wall power do not use the same performance mode on this machine. Re-measured
by alternating the two builds within each repetition, which cancels any drift
over the run, the end-to-end stack is:

| Layer | baseline | with the defaults | speedup | originally reported |
|---|---:|---:|---:|---:|
| q_proj | 285 | 152 | 1.87x | 1.99x |
| fc1 | 212 | 104 | 2.03x | 2.03x |
| fc2 | 2662 | 1642 | 1.62x | 1.76x |

Read the stack as **1.6x to 2.0x**, not 2x. The per-variant rows above are still
the best attribution available, since re-measuring each step interleaved has not
been done, but treat their individual sizes as approximate.

Power state itself turned out to matter little: the same configuration measured
on battery and on wall power differs by under 7%, in both directions, and low
power mode was off throughout. What the block design could not cancel was drift
of any kind, and the drift happened to fall in the flattering direction twice.

The fix costs nothing. `experiments/run_interleaved.sh` takes several variants and
runs the same solves the old protocol did, reordered so every (row, seed) point is
measured for all variants back to back; drift then hits all of them equally. It is
cheaper than running each variant separately, because one process covers every
variant instead of one process per variant. Use `experiments/ab_interleaved.sh`
only when the two sides are different commits, which flags cannot express.

- **A2** replaces per-element reads of device tensors with one transfer. The
  removed-index list cost 113 ms per call at M=768 and 1 ms after the change.
- **A1** skips a pass whose starting point is settled, meaning an earlier pass
  examined it and found nothing and nothing has moved since. Whether a pass finds
  a move depends on the point, not on the random draws, because the pass
  evaluates every candidate and keeps the best. The skipped pass would have drawn
  from the random stream, so this is statistically equivalent rather than
  trajectory identical, and it is judged at equal time.
- **A3** keeps the running best in six device tensors updated by a masked
  comparison, so the host reads the answer once per pass instead of four scalars
  per candidate chunk. Synchronizations fell from 421 to 155 per iteration and
  the selected swap is unchanged.
- **A4** sizes the row tile from a memory budget instead of a fixed 256 rows.
  Operation count fell from 46,900 to 5,200 per iteration.

### A6, incumbent pruning: conditional, and off by default

Selection keeps the candidate with the smallest maximum, so a candidate whose
running maximum has already passed the best one found so far can never win and
its remaining samples are dead work. The comparison is strict, so candidates that
can still tie survive and remain able to win on the sum of squares.

| Configuration | q_proj | fc2 |
|---|---:|---:|
| candidate chunk 512, no pruning | 150 | 1619 |
| candidate chunk 32768, no pruning | 163 | 1941 |
| candidate chunk 512, pruning | 189 | **1350** |
| candidate chunk 32768, pruning | 151 | 2238 |

At a fixed 20 iterations, pruning wins 1.20x on the wide layer and loses 1.26x on
the narrow one. At a fixed time budget it reaches more iterations on every layer:

| Layer | Iterations in 10 s, without | With | Rows worse at equal time |
|---|---:|---:|---:|
| q_proj | 100 | 126 | 0 of 12 |
| fc1 | 174 | 214 | 0 of 12 |
| fc2 | 3 | 6 | 0 of 12 |

The two measures disagree because iteration cost is not stationary: early
iterations do far more local-search work than late ones, so a fixed-iteration
measure over-weights the expensive early ones, where pruning's per-stage cost
falls hardest. Confirmed directly on q_proj: over 20 iterations pruning is 21%
slower per iteration, over 100 iterations 5% slower, and both reach identical
objectives at both lengths. The time budget is what production uses, so that is the measure that
should decide.

It is still off by default, because the margin on narrow layers is inside the
noise of this device and the balance is expected to move at the production sample
count, where the saving grows and the per-stage cost does not. Enable it with
`AMVM_PRUNE_TO_INCUMBENT=1` and measure per layer on the target device.

Raising the candidate chunk is a regression in every combination and is not used.

### A5 and A7, designed and not implemented

- **A5, storing activations transposed.** Candidate evaluation gathers activation
  columns, which are strided in the current layout. On MPS gathers are not the
  bottleneck: `aten::index` accounted for 39 ms of a run whose synchronizations
  accounted for 453 ms. The change is also not free in memory, because a second
  copy of the activation matrix is 3.2 GB at production size for `fc2`, so it
  must replace the layout rather than add to it, which touches every consumer of
  `state.inputs`. Worth doing only if a CUDA profile shows gathers dominating.
- **A7, the L2 term from the Gram matrix.** Sum of squares over all samples can be
  computed from `X'X`, which is shared by every row. Measured on this device, the
  squares accumulation is about 10% of the exact stage: 8.4 ms with it against
  7.6 ms without, for 1,024 candidates over 16,384 samples. The upside is small
  and the float-accumulation risk is real, so it was not implemented.

## Tier B, batched rows

`src/GPU/ALNS/batched.py` holds a group of rows in one tensor program:
`(rows, variables)` weights against `(samples, rows)` residuals, one candidate
list carrying a row tag, and one winner per row chosen by three segment
reductions. Rows that stop improving are masked out of the candidate build.

It does not use the `alns` package. That package drives one instance per Python
loop and owns operator selection and acceptance, which is the structure batching
has to remove. The loop here is destroy, repair, batched swap, hill-climbing
acceptance per row. Tomography and the FIR design keep using the package.

Enable with `AMVM_BATCHED_ROWS=1`; `AMVM_BATCH_SIZE` sets how many rows share one
program. GPTQ starting weights are refused by this path rather than silently
mishandled, because the GPTQ zero-point domain is not implemented here.

**Read the budget before enabling this on CUDA.** A group is given
`seconds * group size`, which is what its rows would have consumed one after
another. That matches a single-device sequential run. It does not match the
production setup, where forty rows run at once for `seconds` each, so a 64-row
group would occupy roughly forty times the wall clock of the current fan-out.
Enabling the flag there without changing the budget is a large slowdown, not a
speedup.

The batched path also prunes against the acceptance bound, which is the strategy
measured at 0.15% of work removed, and it pays a compaction synchronization per
stage to do it. Its parity result is therefore slightly pessimistic; pruning
against each row's incumbent, as the single-row path does, would remove that dead
cost. This does not change the off-by-default decision.

It reaches parity and is not enabled. At equal total time, where the per-row path
gets the production budget per row and the batched path gets the sum for the
group:

| Instance | Rows | Per-row path | Batched path | Rows better |
|---|---:|---|---|---:|
| q_proj | 16 | 161.3 s, 0.654913 | 165.1 s, 0.659705 | 5 of 16 |
| q_proj | 64 | 646.5 s, 0.643139 | 640.1 s, 0.654430 | 28 of 64 |
| fc2 | 16 | 189.5 s, 0.043478 | 160.2 s, 0.043536 | 7 of 16 |

The premise of batching is that one row does not fill the device. After the Tier
A changes that premise no longer holds here: the per-row program already keeps
this device busy, and what batching adds is the Python and launch cost Tier A had
already removed. Batch size moves the result the right way without crossing over,
from 31% of rows won at 16 rows to 44% at 64.

Two defects in this path were found by benchmarking rather than by tests, and
both are now covered. Level values were read positionally instead of by row
index, which is invisible while all rows share one domain and all rows improve.
A row with no admissible candidate was treated as a row whose candidates all tie,
because its best key and their keys were both infinite, so it was handed a move
it should not have taken.

A separate finding from these runs: the per-row path overran its time budget by
18% on `fc2`, because the stopping rule is only checked between iterations and an
`fc2` iteration is long. At the production sample count an `fc2` iteration would
be far longer than the whole per-row budget, so the budget would stop meaning
much. Worth deciding deliberately rather than discovering in a run.

## What to merge

| Change | Recommendation | Evidence |
|---|---|---|
| A2 host-side index gathering | merge, on by default | exact; 1.29x, 1.38x, 1.13x |
| A1 settled-pass skip | merge, on by default | 0 of 36 paired rows worse at equal time |
| A3 device-resident swap | merge, on by default | identical selection; synchronizations 421 to 155 |
| A4 budget-derived row tile | merge, on by default | identical objective; operations 46,900 to 5,200 |
| A6 incumbent pruning | ON by default since 2026-09-18, paired with residual ordering | alone it is 0.84x and 0.79x, i.e. slower than no pruning, on the narrow layers |
| Residual-ordered rows | merge, on by default | exact on real layers; 1.25x, 1.39x, 1.65x over no pruning; no paired row worse |
| Local-search deadline | merge, on by default | one 10s solve had been running 180s+ under the tie-break policy |
| Acceptance policy | keep `linf_l2_nonincrease` | no policy wins everywhere; removing the gate buys iterations and loses quality |
| Budget split across replicas | do not implement | worse or neutral everywhere; +24% on fc2 at five replicas |
| A7 Gram-matrix L2 | do not implement yet | the term it removes is 10% of the stage it runs in |
| A5 transposed activations | do not implement yet | gathers are not the bottleneck here, and a second copy is 3.2 GB |
| Tier B batched rows | merge behind a flag, off | parity at equal time on this device |

The four defaults together are 1.87x, 2.03x and 1.62x per iteration when measured
interleaved, and at equal time they reach a lower infinity norm on every layer
with no paired row worse than the baseline in any of 36 comparisons.

The equal-time comparisons were also run in blocks, and they are not corrected
here, but their direction is safe. The variants ran in ladder order while the
battery drained, so any drift-induced slowdown handicapped the later variants,
and the later variants are the ones that won. The result is conservative rather
than flattered.

The flags are the mechanism for keeping this decidable later: every variant lives
on one commit, `experiments/variants.sh` names the cumulative stack, and
`experiments/run_protocol.sh` runs the same commands for each, so re-deciding on
CUDA means rerunning the protocol rather than rebuilding the comparison.

## Open checks that need CUDA

None of the following could be run on this machine. They are ordered by how much
they would change the conclusions above.

1. **Is the production GPU idle or saturated?** This decides whether the
   synchronization work above transfers at all. Ten rows per GPU run as separate
   processes, which may already fill the device.
   ```bash
   nvidia-smi dmon -s u -d 5 -o T
   ```
   Low streaming-multiprocessor utilization means the workers are waiting and
   these changes transfer directly. High utilization means process concurrency
   already fills the device and kernel efficiency matters more than launches.

2. **Enumerate the remaining synchronization sites**, which has no MPS
   equivalent. Add to one worker and read the stack traces:
   ```python
   torch.cuda.set_sync_debug_mode("warn")
   ```

3. **Re-tune the row tile at the production sample count.** The budget rule gives
   43,690 rows per tile at N=262,144 against the fixed 256. Sweep the budget:
   ```bash
   for mb in 64 256 1024; do AMVM_SKIP_SETTLED_LOCAL_SEARCH=1 AMVM_DEVICE_RESIDENT_SWAP=1 \
     AMVM_BUDGET_ROW_TILE=1 AMVM_SWAP_MEMORY_BUDGET_MB=$mb \
     python experiments/bench_row_solve.py run --mode speed --n 262144 \
     --instance experiments/data/model_decoder_layers_0_fc2.pt --label budget-$mb \
     --out experiments/results/budget-$mb/fc2_speed.json; done
   ```

4. **Re-decide A6 at the production sample count.** Its saving scales with sample
   count while its cost does not, so the layer threshold measured here is likely
   to move. Run the A6 table above with `--n 262144`.

5. **Peak memory.** Nothing here measures CUDA peak allocation. Wrap a row solve
   in `torch.cuda.max_memory_allocated()` and confirm the budget rule bounds it.

6. **Does batching beat process-level concurrency?** The comparison that matters
   is the batched path against ten worker processes per GPU, not against one
   sequential loop, since that is what production runs today.
   ```bash
   AMVM_BATCHED_ROWS=1 python experiments/bench_batched_rows.py \
     --instance experiments/data/model_decoder_layers_0_fc2.pt --rows 64 --n 262144
   ```

7. **The NVIDIA MPS daemon**, which lets the existing worker processes overlap
   kernels with no code change. Worth trying before any of the above if step 1
   shows the device idle.

8. **Whole-matrix quality and perplexity.** Every measurement here is per row on
   three layers. Nothing has been run end to end, so no claim about model quality
   is supported yet.
