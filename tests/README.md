# Regression tests

This suite exercises the real PyTorch solver on tiny, deterministic problems.
It adds tests and CI without changing the optimization algorithm. No model
downloads, customer data, Gurobi license, or GPU are required for CPU CI.

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

## Known defects are executable, not skipped

At the reviewed baseline `dee5ce6b921475c9ee7f3eef1b150823d513a476`, CPU validation
on Python 3.12 produces **11 passed, 12 xfailed** across 23 test cases.
The 12 expected failures cover 10 distinct defect IDs below.

Each known defect has a `known_bug` marker and a strict `xfail` restricted to
`AssertionError`. The test still runs. An unexpected exception fails CI; an
unexpected pass also fails CI so that the obsolete marker must be removed.
An expected failure does not certify that the corresponding solver behavior is
correct. It documents existing debt while allowing new regressions to block PRs.

To expose every known defect as an ordinary failure:

```bash
python -m pytest --runxfail
```

That command is expected to exit nonzero until the bugs are fixed. To audit only
the documented defects, use `python -m pytest -m known_bug --runxfail`.
When fixing a defect, remove its `xfail` and `known_bug` decorators in the same PR
and retain the correctness assertion as a permanent regression test.

| ID | Required behavior | Test module |
| --- | --- | --- |
| B01 | Greedy repair evaluates each move against the current residual | `test_operators.py` |
| B02 | Nonuniform single-variable scoring uses the actual level difference | `test_state.py` |
| B03 | Greedy repair and 1-OPT preserve fixed outlier residuals | `test_operators.py` |
| B04 | Tomography keeps its prescribed grey-level domain | `test_integration.py` |
| B05 | Candidate states own independent mutable change queues | `test_state.py` |
| B06 | Random destroy selects at least one variable in a small nonempty instance | `test_operators.py` |
| B07 | Destroying a requested fraction selects distinct variables | `test_operators.py` |
| B08 | Unexpected runtime errors propagate out of both local-search entry points | `test_operators.py` |
| B09 | Quantization retries only failed rows | `test_integration.py` |
| B10 | Float64 residual updates do not round their update buffer to float32 | `test_state.py` |

Passing tests cover initial residual/objective caches, committed moves and their
inverses, read-only move evaluation, candidate tensor independence, swap
evaluation and rollback, safe row screening on a small example, and a short
end-to-end solve checked against direct matrix multiplication.

The `policy` test deliberately characterizes the current L2 rejection rule. It
does not assert that this is the right policy for DMMV. If the policy changes,
replace that test with tests of the newly specified behavior.

## CI

`.github/workflows/tests.yml` runs CPU tests on Python 3.10 and 3.12 for pull
requests and pushes to `main`, `test/**`, and `fix/**`. It uploads a JUnit report
even when tests fail. The workflow has read-only repository permissions and
does not deploy, publish a package, or change branch protection.

Manual workflow dispatch offers `audit_known_bugs=true`, equivalent to
`--runxfail`. GitHub only exposes the manual-dispatch UI once this workflow is
present on the default branch. Maintainers can make the two CPU jobs required
checks after merging.

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

Perplexity, algorithmic quality across seeds, complete RNG ownership, full
application reproducibility, fallback policy, paper/code agreement, and
candidate-cap tradeoffs remain outside this unit suite. Tests explicitly seed
Python, NumPy and PyTorch for repeatability; that does not establish a complete
seed API in the production solver.
