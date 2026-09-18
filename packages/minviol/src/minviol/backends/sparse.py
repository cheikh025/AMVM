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

    def apply_delta(self, y, instances, left, right, delta):
        """Add each instance's move to its column of ``y``, touching only its support.

        Densifying the column first would put an ``O(constraints)`` allocation on
        every accepted move, which on a sparse matrix is the cost the whole
        backend exists to avoid -- and it would be paid once per move rather than
        once per pass.
        """
        move, rows, accumulated = self._column_difference(left, right)
        if not len(move):
            return
        # The column difference is assembled before scaling, so a constraint
        # touched by both halves of a swap gets one combined value and the
        # arithmetic is grouped exactly as the candidate's score was.
        y.index_put_((rows, instances[move]), delta[move] * accumulated, accumulate=True)

    def rows_of(self, constraints):
        """``A[constraints, :]`` densified as ``(len(constraints), variables)``.

        One row per instance, so this is a thin block however large the matrix is.
        """
        return self.dense_rows(constraints)

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

    def lookup(self, cols, rows):
        """Locate ``A[rows, cols]``: ``(found, position)`` into the stored entries.

        A binary search into the globally sorted column-major key, so the whole
        test is a couple of vectorized ops regardless of how ragged the columns
        are, and needs no dense row slab.
        """
        key = (cols * self.shape[0] + rows).reshape(-1)
        position = torch.searchsorted(self.key, key).clamp_max(max(0, len(self.key) - 1))
        found = self.key[position] == key
        return found.reshape(cols.shape), position.reshape(cols.shape)

    def contains(self, cols, rows):
        """Elementwise test of whether ``A[rows, cols]`` is stored."""
        return self.lookup(cols, rows)[0]

    def _independent_steps(self, left, right, delta, delta_right):
        """``(owner, constraint, step)`` for a compound move's two independent steps.

        A swap's two steps are equal and opposite, so its column difference can be
        assembled before either step scales it -- which is what keeps the sparse
        and dense backends bitwise equal there. A compound move's steps do not
        factor, so each column is scaled by its own step and the two are summed
        per ``(candidate, constraint)``. Constraints touched by both columns
        appear once, with the sum.
        """
        owner, rows, values = self._gather_columns(left)
        contribution = delta[owner] * values
        owner_r, rows_r, values_r = self._gather_columns(right)
        owner = torch.cat([owner, owner_r])
        rows = torch.cat([rows, rows_r])
        contribution = torch.cat([contribution, delta_right[owner_r] * values_r])
        if not len(owner):
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.empty(0, dtype=self.dtype, device=self.device)

        key = owner * self.shape[0] + rows
        unique_key, inverse = torch.unique(key, return_inverse=True)
        accumulated = torch.zeros(len(unique_key), dtype=self.dtype, device=self.device)
        accumulated.scatter_add_(0, inverse, contribution)
        return (unique_key // self.shape[0], unique_key % self.shape[0], accumulated)

    def _column_difference(self, left, right):
        """Union of the two columns' supports, as ``(owner, constraint, value)``.

        For a swap the value is ``A[i, left] - A[i, right]``, with a missing entry
        read as zero. Each ``(owner, constraint)`` appears exactly once by
        construction: the left column contributes all of its own entries, already
        carrying the subtraction where the right column overlaps, and the right
        column contributes only what the left one does not have.

        Built this way rather than by concatenating both columns and merging with
        ``torch.unique``. That merge sorts millions of keys for what is really two
        binary searches, and the sort was measured to cost the swap pass more than
        the sparsity saved it.
        """
        owner, rows, values = self._gather_columns(left)
        if right is None:
            return owner, rows, values

        overlap, position = self.lookup(right[owner], rows)
        # a - b, not a + (-b) grouped through an accumulator: both are the same
        # IEEE operation, and this one needs no accumulator at all.
        values = values - torch.where(overlap, self.values[position],
                                      torch.zeros_like(values))

        owner_r, rows_r, values_r = self._gather_columns(right)
        if len(owner_r):
            already, _ = self.lookup(left[owner_r], rows_r)
            fresh = ~already
            owner = torch.cat([owner, owner_r[fresh]])
            rows = torch.cat([rows, rows_r[fresh]])
            values = torch.cat([values, -values_r[fresh]])
        return owner, rows, values

    # -- candidate scoring -------------------------------------------------

    def screen(self, batch, tag, left, right, delta, bound, screen_rows, tile):
        """No screening stage: the exact stage is already proportional to ``nnz``.

        Screening exists to keep candidates away from a full pass over every
        constraint. Here a candidate never costs that, so screening against K
        constraints would cost more than the answer it is protecting.
        """
        return torch.ones(len(tag), dtype=torch.bool, device=self.device)

    def _chunk_bounds(self, left, right, options):
        """Split the candidate list so each chunk's flat gather fits the budget.

        Chunking by candidate count is not enough: what a candidate costs is its
        column's support, and a wide column can be thousands of entries. On a 5%
        dense matrix one block of 3,500 swap candidates expands to 17 million
        entries, which is where the intermediates stop fitting anywhere sensible.
        The split therefore follows the cumulative entry count, not the index.
        """
        counts = self.col_ptr[left + 1] - self.col_ptr[left]
        if right is not None:
            counts = counts + (self.col_ptr[right + 1] - self.col_ptr[right])
        # Entries carry an index, a value, a key and an accumulator, plus what
        # torch.unique needs to sort them.
        per_entry = 8 + 4 * 4
        budget = max(1, options.memory_budget_mb * 1024 * 1024 // per_entry)
        running = torch.cumsum(counts, 0)

        bounds, start = [], 0
        total = int(running[-1].item())
        while start < len(left):
            consumed = 0 if start == 0 else int(running[start - 1].item())
            if total - consumed <= budget:
                bounds.append((start, len(left)))
                break
            stop = int(torch.searchsorted(running, consumed + budget).item())
            stop = min(max(stop, start + 1), len(left))   # always make progress
            bounds.append((start, stop))
            start = stop
        return bounds

    def exact_scores(self, batch, tag, left, right, delta, prune_bound, screen_rows,
                     options, counters, delta_right=None):
        """Exact maximum violation and sum of squares per candidate."""
        n_candidates = len(tag)
        maxima = torch.empty(n_candidates, dtype=self.dtype, device=self.device)
        squares = torch.empty_like(maxima)
        if not n_candidates:
            return maxima, squares
        for start, stop in self._chunk_bounds(left, right, options):
            piece = slice(start, stop)
            counters.bump("sparse_chunks")
            maxima[piece], squares[piece] = self._score_block(
                batch, tag[piece], left[piece],
                None if right is None else right[piece], delta[piece],
                screen_rows=screen_rows, counters=counters,
                delta_right=None if delta_right is None else delta_right[piece])
        return maxima, squares

    def _score_block(self, batch, tag, left, right, delta, screen_rows, counters,
                     delta_right=None):
        untouched = self._untouched_max(batch, tag, left, right, screen_rows, counters)
        touched_max, delta_l2 = self._touched(batch, tag, left, right, delta, counters,
                                              delta_right)
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

    def _touched(self, batch, tag, left, right, delta, counters, delta_right=None):
        """Maximum violation over the touched constraints, and the change in L2."""
        n_candidates = len(tag)
        if delta_right is None:
            cand, row, accumulated = self._column_difference(left, right)
            step = None if not len(cand) else delta[cand] * accumulated
        else:
            cand, row, step = self._independent_steps(left, right, delta, delta_right)
        if not len(cand):
            return (torch.full((n_candidates,), NEG_INF, dtype=self.dtype,
                               device=self.device),
                    torch.zeros(n_candidates, dtype=self.dtype, device=self.device))

        instance = tag[cand]
        counters.bump("constraint_candidate_products", len(cand))

        y_old = batch.y[row, instance]
        lower, upper = batch.lower[row, instance], batch.upper[row, instance]
        scale = None if batch.row_scale is None else batch.row_scale[row, instance]
        v_old = viol.violation(y_old, lower, upper)
        v_new = viol.violation(y_old + step, lower, upper)
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
