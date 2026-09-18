"""Matrix backends.

A backend owns the constraint matrix and every kernel that touches it. The
engine owns the search. The split is drawn so that a backend is asked to score a
whole candidate list at once, never one candidate at a time: a per-candidate
interface would reintroduce the host synchronization that made this solver slow
in the first place.
"""

from .dense import DenseMatrix
from .sparse import SparseMatrix

__all__ = ["DenseMatrix", "SparseMatrix"]
