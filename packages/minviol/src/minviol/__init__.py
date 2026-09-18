"""minviol -- find a point satisfying a large constraint system, on a GPU.

The solver minimizes the largest violation of ``lower <= A x <= upper`` over a
discrete domain. An objective of zero is a feasible point. It is a heuristic:
there is no certificate, and a nonzero answer means "not found", never
"infeasible".
"""

from .api import Result, solve, solve_batch
from .backends import DenseMatrix
from .counters import Counters
from .options import Budget, Options
from .violation import violation

__all__ = ["solve", "solve_batch", "Result", "Budget", "Options", "DenseMatrix",
           "Counters", "violation"]
__version__ = "0.1.0.dev0"
