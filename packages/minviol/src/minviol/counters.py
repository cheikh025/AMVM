"""Solver work counters.

Counting is a host-side dictionary update, so it never synchronizes with the
device and can stay on in production. Unlike the process-global counter this was
lifted from, a ``Counters`` instance belongs to one solve: a library cannot
assume one process means one problem, and two solves running in the same process
must not add their work together.
"""

import collections


class Counters:
    """Work counts for a single solve."""

    def __init__(self):
        self._counts = collections.Counter()

    def bump(self, name: str, amount: int = 1) -> None:
        self._counts[name] += amount

    def snapshot(self) -> dict:
        return dict(self._counts)

    def __getitem__(self, name: str) -> int:
        return self._counts[name]

    def __repr__(self) -> str:
        return f"Counters({dict(self._counts)!r})"
