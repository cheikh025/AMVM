"""Process-local solver counters.

Counting is a host-side dictionary update, so it never synchronizes with the
device and can stay on in production. Each row runs in its own process, so the
counts describe exactly one row solve. ``snapshot`` feeds run manifests and
ledger entries; without these numbers a speed comparison cannot be interpreted.
"""

import collections

COUNTS = collections.Counter()


def bump(name: str, amount: int = 1) -> None:
    COUNTS[name] += amount


def reset() -> None:
    COUNTS.clear()


def snapshot() -> dict:
    return dict(COUNTS)
