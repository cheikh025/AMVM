"""minviol -- find a point satisfying a large constraint system, on a GPU.

The solver minimizes the largest violation of ``lower <= A x <= upper`` over a
discrete domain. An objective of zero is a feasible point. It is a heuristic:
there is no certificate, and a nonzero answer means "not found", never
"infeasible".
"""

from .api import Result, solve, solve_batch
from .backends import DenseMatrix, SparseMatrix
from .counters import Counters
from .options import Budget, Options
from .violation import violation

__all__ = ["solve", "solve_batch", "Result", "Budget", "Options", "DenseMatrix", "SparseMatrix",
           "Counters", "violation"]

# Read from the installed distribution rather than written here, so there is one
# place a version can be wrong. Hardcoding it had already drifted: the wheel said
# 0.1.0 and the import said 0.1.0.dev0.
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("minviol")
except (ImportError, PackageNotFoundError):  # running from a source tree
    __version__ = "0.0.0+unknown"
