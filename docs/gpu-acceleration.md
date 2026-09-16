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
read the tables with `experiments/summarize_results.py`.

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

Median milliseconds per iteration, 20 iterations, four rows, profiler off. The
stack is 2.0x on q_proj, 2.0x on fc1 and 1.8x on fc2 against the baseline.

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
| candidate chunk 512, no pruning | 146 | 1616 |
| candidate chunk 32768, no pruning | 158 | 1893 |
| candidate chunk 512, pruning | 170 | **1328** |
| candidate chunk 32768, pruning | 151 | 2216 |

Pruning wins 1.22x on the wide layer and loses 1.16x on the narrow one. The
saving grows with the number of candidates per level pair and with the sample
count, while the cost is a fixed synchronization per stage, so wide layers pay it
off and narrow ones do not. Wide layers also dominate total matrix time, so
enabling it per layer is worth measuring on the target device. It is off by
default and enabled with `AMVM_PRUNE_TO_INCUMBENT=1`.

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
