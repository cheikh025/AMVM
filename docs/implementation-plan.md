# Correctness and validation plan

Baseline: `68501f61302505f6b124df1516945c40bd935d86` (clean working tree).

Execute in this order, keeping regression checks after each stage:

1. Explicit domain input, physical move deltas, read-only evaluation and immediate residual updates. Check against direct matrix multiplication, including nonuniform domains and tomography.
2. One fixed-variable mask and configurable acceptance (`linf`, `linf_l2_tiebreak`, `linf_l2_nonincrease`). Check tiny exhaustive neighborhoods, copied-state ownership and destroy operators.
3. Owned RNGs and seed API, unexpected-error propagation, structured row statuses, failed-row-only retries, mode-preserving explicit fallbacks and run metadata.
4. Tile both swap groups and filtering/evaluation rows; evaluate all survivors in bounded chunks. Remove debug synchronization/cache flushing. Verify equivalence against exhaustive small cases; measure CPU behavior here and provide CUDA measurement commands.
5. Align implementation documentation and report evidence gaps. Additional algorithm changes need equal-time, multi-seed ablations on the corrected baseline; do not promote speculative changes without measurements.

No manuscript source is present in this checkout. Paper claims and historical tomography results require a provenance audit and reruns before reuse.
