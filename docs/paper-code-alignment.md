# Paper and implementation alignment

No manuscript source is present in this checkout, so the paper cannot be edited
here. Apply the following corrections to any manuscript that describes this
implementation, and audit the provenance of previously reported results.

| Topic | Supported implementation statement | Unsupported statement to remove unless another artifact is identified |
|---|---|---|
| Execution | Eager PyTorch with bounded tensor tiles and Python control flow | Fused CUDA kernels throughout |
| Greedy repair | Removed variables are visited sequentially; each accepted move immediately updates the residual | Bulk-vectorized globally greedy repair |
| Local search | `LS_op="S"` runs adjacent-level swaps; `LS_op="W"` runs swaps followed by adjacent one-variable moves | Both move classes run in every iteration |
| Swap selection | The current bounded implementation evaluates every survivor and selects the best adjacent-level pair; wider level differences require `use_all_delta_q` | Unqualified global best over every pair and every level difference |
| Objective policy | Pure infinity norm is the default. L2 is available as an explicit tie-break or nonincrease constraint | An implicit or universally applied L2 restriction |
| Domains | The solver accepts an explicit physical discrete domain; tomography passes `[0, MAX_RANGE]` levels directly | Domain inferred from reconstruction extrema |
| Memory | Variable groups, screening rows, and exact evaluation are tiled; tile sizes bound peak intermediates | Unbounded `rows x group1 x group2` materialization |

The CPU microbenchmark in `experiments/results/swap_tiles_cpu_default.json`
matches the dense best candidate and reduces its filter-tensor bound by 4x for
the measured 32 by 512 by 512 case, while running 2.54x slower. For 100 rows
and groups of 2,048, a 256-variable tile bounds that float32 filter tensor at
25 MiB instead of 1.56 GiB. CUDA peak allocation and time still need measurement.

Tomography results produced before the explicit-domain change may solve over
levels derived from SART extrema. Do not reuse those numbers without rerunning
from a recorded commit and configuration and verifying that every returned
value belongs to the prescribed grey-level domain.
