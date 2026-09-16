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

## Running summary

| # | Date | Title | Layer | Decision | Correctness | Memory | Runtime |
|---|---|---|---|---|---|---|---|
| 2 | 2026-09-15 | Bounded exhaustive swap evaluation | Infrastructure | KEEP | exact match | 4x smaller filter tensor in measured default case | 2.54x slower on CPU microbenchmark |
| 1 | 2026-09-15 | Explicit domains and physical move updates | Evaluator/Search | KEEP | 27/27 tests pass | neutral | full benchmark unavailable |

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
