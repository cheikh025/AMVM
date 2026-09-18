"""Sparse constraint matrix.

When ``A`` is sparse, changing one variable perturbs only the constraints in that
variable's column support, so a candidate costs ``nnz(col j)`` instead of every
constraint. The obstacle is that the objective is a *global* maximum: a
candidate's score needs the largest violation among the constraints it did not
touch, and computing that directly is the whole-matrix reduction the saving was
supposed to avoid.

Keeping the top-K violations removes it. Writing ``T`` for the K largest
violations, ``S`` for the touched constraints and ``U`` for the untouched ones:

    max(U) == max(T \\ S)       whenever  T \\ S is non-empty

because everything outside ``T`` is at most everything inside it, so the largest
untouched constraint, if it were outside ``T``, would tie any untouched member of
``T``. Carrying ``tau``, the (K+1)-th violation, closes the remaining case:
``max(max(T \\ S), tau)`` is always an upper bound and is exact whenever
``T \\ S`` is non-empty. Since ``nnz(col j) < K`` forces that set to be
non-empty, choosing ``K`` above the widest column makes it exact everywhere --
and the count of inexact candidates is reported rather than assumed.

Storage is deliberately not ``torch.sparse_csr``/``_csc``: compressed sparse
tensors are not implemented on the MPS backend, so a solver built on them would
be CUDA and CPU only. The structure is carried as ordinary tensors instead,
which is also closer to what the kernels want -- column gathers and segment
reductions, not a general sparse matmul.
"""

import torch

from .. import violation as viol

NEG_INF = float("-inf")


class SparseMatrix:
    """``(constraints, variables)`` sparse matrix held as plain index tensors."""

    kind = "sparse"

    def __init__(self, indices, values, shape, *, device=None, dtype=None):
        """``indices`` is ``(2, nnz)`` of (constraint, variable); duplicates are refused."""
        device = device or values.device
        dtype = dtype or values.dtype
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        n_constraints, n_variables = self.shape

        rows = indices[0].to(device=device, dtype=torch.long)
        cols = indices[1].to(device=device, dtype=torch.long)
        values = values.to(device=device, dtype=dtype)
        if bool(((rows < 0) | (rows >= n_constraints)).any()) or \
           bool(((cols < 0) | (cols >= n_variables)).any()):
            raise ValueError("sparse indices fall outside the declared shape")

        # Column-major order, and the key that makes membership a binary search.
        key = cols * n_constraints + rows
        order = torch.argsort(key)
        self.key = key[order]
        self.row_idx = rows[order]
        self.col_of = cols[order]
        self.values = values[order]

        # A duplicated entry is invisible to a from-scratch A @ x, which coalesces
        # it, but is applied twice by an incremental update. That shows up as slow
        # drift rather than a failure, so it is refused at the door.
        if len(self.key) > 1 and bool((self.key[1:] == self.key[:-1]).any()):
            raise ValueError("sparse matrix has duplicate (constraint, variable) entries; "
                             "coalesce it first")

        counts = torch.bincount(self.col_of, minlength=n_variables)
        self.col_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                  counts.cumsum(0)])
        self.max_nnz = int(counts.max()) if len(counts) else 0
        self.nnz = int(len(self.key))

        # COO carries the matrix-vector product. It is the one sparse tensor type
        # MPS implements, and this is the only place a general product is needed.
        # Invariants are checked above -- indices are in range and deduplicated --
        # so the per-construction check is opted out of deliberately rather than
        # left to a warning.
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            self._coo = torch.sparse_coo_tensor(
                torch.stack([self.row_idx, self.col_of]), self.values, self.shape,
                device=device, dtype=dtype).coalesce()

    @classmethod
    def from_dense(cls, A):
        index = A.nonzero(as_tuple=False).T
        return cls(index, A[index[0], index[1]], A.shape, device=A.device, dtype=A.dtype)

    @classmethod
    def from_torch_sparse(cls, A):
        A = A.coalesce() if A.layout == torch.sparse_coo else A.to_sparse_coo().coalesce()
        return cls(A.indices(), A.values(), A.shape, device=A.device, dtype=A.dtype)

    # -- basic products ----------------------------------------------------

    def matvec(self, x_phys):
        return torch.sparse.mm(self._coo, x_phys.T)

    def columns(self, cols):
        """``A[:, cols]`` densified as ``(constraints, len(cols))``."""
        out = torch.zeros(self.shape[0], len(cols), device=self.device, dtype=self.dtype)
        cand, rows, vals = self._gather_columns(cols)
        out[rows, cand] = vals
        return out

    def dense_rows(self, rows):
        out = torch.zeros(len(rows), self.shape[1], device=self.device, dtype=self.dtype)
        position = torch.full((self.shape[0],), -1, dtype=torch.long, device=self.device)
        position[rows] = torch.arange(len(rows), device=self.device)
        where = position[self.row_idx]
        live = where >= 0
        out[where[live], self.col_of[live]] = self.values[live]
        return out

    # -- ragged gathers ----------------------------------------------------

    def _gather_columns(self, cols):
        """Flatten the supports of ``cols`` into ``(owner, constraint, value)``.

        One batched expansion rather than a Python loop over candidates: the
        number of statements must not grow with the number of candidates.
        """
        counts = self.col_ptr[cols + 1] - self.col_ptr[cols]
        total = int(counts.sum().item())
        if total == 0:
            empty_i = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_i, empty_i, torch.empty(0, dtype=self.dtype, device=self.device)
        owner = torch.repeat_interleave(torch.arange(len(cols), device=self.device), counts)
        starts = torch.cumsum(counts, 0) - counts
        within = torch.arange(total, device=self.device) - starts[owner]
        flat = self.col_ptr[cols][owner] + within
        return owner, self.row_idx[flat], self.values[flat]

    def contains(self, cols, rows):
        """Elementwise test of whether ``A[rows, cols]`` is stored.

        A binary search into the globally sorted column-major key, so the whole
        test is two vectorized ops regardless of how ragged the columns are.
        """
        key = cols * self.shape[0] + rows
        position = torch.searchsorted(self.key, key.reshape(-1)).clamp_max(len(self.key) - 1)
        return (self.key[position] == key.reshape(-1)).reshape(key.shape)

    # -- candidate scoring -------------------------------------------------

    def screen(self, batch, tag, left, right, delta, bound, screen_rows, tile):
        """No screening stage: the exact stage is already proportional to ``nnz``.

        Screening exists to keep candidates away from a full pass over every
        constraint. Here a candidate never costs that, so screening against K
        constraints would cost more than the answer it is protecting.
        """
        return torch.ones(len(tag), dtype=torch.bool, device=self.device)

    def exact_scores(self, batch, tag, left, right, delta, prune_bound, screen_rows,
                     options, counters):
        """Exact maximum violation and sum of squares per candidate."""
        n_candidates = len(tag)
        maxima = torch.empty(n_candidates, dtype=self.dtype, device=self.device)
        squares = torch.empty_like(maxima)
        tile = max(1, options.candidate_tile)
        for start in range(0, n_candidates, tile):
            piece = slice(start, start + tile)
            maxima[piece], squares[piece] = self._score_block(
                batch, tag[piece], left[piece],
                None if right is None else right[piece], delta[piece],
                screen_rows=screen_rows, counters=counters)
        return maxima, squares

    def _score_block(self, batch, tag, left, right, delta, screen_rows, counters):
        untouched = self._untouched_max(batch, tag, left, right, screen_rows, counters)
        touched_max, delta_l2 = self._touched(batch, tag, left, right, delta, counters)
        return torch.maximum(untouched, touched_max), batch.l2[tag] + delta_l2

    def _untouched_max(self, batch, tag, left, right, screen_rows, counters):
        """Largest violation among the constraints this candidate does not touch."""
        rows = screen_rows.indices[tag]                      # (candidates, K)
        values = screen_rows.values[tag]                     # (candidates, K)
        touched = self.contains(left[:, None].expand_as(rows), rows)
        if right is not None:
            touched = touched | self.contains(right[:, None].expand_as(rows), rows)

        usable = (~touched).any(dim=1)
        counters.bump("untouched_max_inexact", int((~usable).sum()))
        masked = torch.where(touched, torch.full_like(values, NEG_INF), values)
        # tau, the (K+1)-th violation, bounds everything outside the kept set, so
        # falling back on it stays sound when a candidate covers all of T.
        return torch.maximum(masked.max(dim=1).values, screen_rows.tau[tag])

    def _touched(self, batch, tag, left, right, delta, counters):
        """Maximum violation over the touched constraints, and the change in L2."""
        n_candidates = len(tag)
        owner, rows, column = self._gather_columns(left)
        if right is not None:
            owner_r, rows_r, vals_r = self._gather_columns(right)
            owner = torch.cat([owner, owner_r])
            rows = torch.cat([rows, rows_r])
            column = torch.cat([column, -vals_r])

        if not len(owner):
            return (torch.full((n_candidates,), NEG_INF, dtype=self.dtype,
                               device=self.device),
                    torch.zeros(n_candidates, dtype=self.dtype, device=self.device))

        # A constraint touched by both columns of a swap appears twice, with one
        # column entry each. Summing per (candidate, constraint) before evaluating
        # is what keeps the two halves from being scored as separate moves.
        key = owner * self.shape[0] + rows
        unique_key, inverse = torch.unique(key, return_inverse=True)
        accumulated = torch.zeros(len(unique_key), dtype=self.dtype, device=self.device)
        accumulated.scatter_add_(0, inverse, column)

        cand = unique_key // self.shape[0]
        row = unique_key % self.shape[0]
        instance = tag[cand]
        counters.bump("constraint_candidate_products", len(unique_key))

        # The step multiplies the assembled column difference, not each half
        # separately. That is the grouping the dense backend uses, and an
        # algebraically equal regrouping would put the two backends a unit in the
        # last place apart on every swap.
        y_old = batch.y[row, instance]
        lower, upper = batch.lower[row, instance], batch.upper[row, instance]
        scale = None if batch.row_scale is None else batch.row_scale[row, instance]
        v_old = viol.violation(y_old, lower, upper)
        v_new = viol.violation(y_old + delta[cand] * accumulated, lower, upper)
        if scale is not None:
            v_old, v_new = v_old * scale, v_new * scale

        touched_max = torch.full((n_candidates,), NEG_INF, dtype=self.dtype,
                                 device=self.device)
        touched_max.scatter_reduce_(0, cand, v_new, reduce="amax", include_self=True)

        # The untouched constraints keep their squares, so the sum only has to be
        # corrected where the point actually changed.
        delta_l2 = torch.zeros(n_candidates, dtype=self.dtype, device=self.device)
        delta_l2.scatter_add_(0, cand, v_new.square() - v_old.square())
        return touched_max, delta_l2
