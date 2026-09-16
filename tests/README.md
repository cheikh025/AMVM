# Regression tests

This suite exercises the real PyTorch solver on tiny, deterministic problems.
No model downloads, customer data, Gurobi license, or GPU are required for CPU CI.

## Run locally

From the repository root, with Python 3.10 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pip check
python -m pytest
```

PyTorch matches the version documented by the project. The direct test
dependencies are pinned; this is not a complete transitive dependency lockfile.

## Correctness coverage

The current CPU suite has **27 passing tests and no expected failures**. It
retains the original counterexamples for stale greedy residuals, nonuniform
level deltas, fixed outliers, tomography domain handoff, copied-state ownership,
small destroy neighborhoods, runtime-error propagation, failed-row-only retries,
and float64 residual updates. It also checks explicit L2 policy selection and
bounded nonuniform swap evaluation against direct matrix multiplication.

## CI

`.github/workflows/tests.yml` runs CPU tests on Python 3.10 and 3.12 for pull
requests and pushes to `main`, `test/**`, and `fix/**`. It uploads a JUnit report
even when tests fail. The workflow has read-only repository permissions and
does not deploy, publish a package, or change branch protection.

GitHub exposes manual workflow dispatch once this workflow is present on the
default branch. Maintainers can make the two CPU jobs required checks after merging.

## GPU checks and coverage limits

On a machine with a compatible CUDA-enabled PyTorch installation, reuse the
state/operator/swap tests on the GPU:

```bash
AMVM_TEST_DEVICE=cuda python -m pytest tests/test_state.py tests/test_operators.py tests/test_swaps.py
```

Requesting CUDA without a working CUDA device fails rather than silently
skipping the tests. CPU CI does not validate GPU latency, synchronization cost,
out-of-memory recovery, or peak allocation. Those require hardware benchmarks;
there are intentionally no flaky wall-clock thresholds or giant allocations
in the unit suite.

The tomography test substitutes only the unused ASTRA import and executes the
real `run_ALNS` wrapper with zero search iterations. It verifies domain handoff,
not tomography reconstruction quality. The retry test substitutes the worker
dispatcher and uses real HDF5 I/O in an isolated temporary directory; it does
not test multiprocessing or GPU scheduling.

Perplexity, algorithmic quality across seeds, real multiprocessing failures,
CUDA peak allocation, tomography reconstruction quality, and paper/code
agreement remain outside this unit suite. Production runs now record the base
seed, acceptance policy, fallback policy, fallback rows, row statuses, and commit.
