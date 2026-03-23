from __future__ import annotations

import numpy as np


"""
Deterministic random baseline.

This is a fast, always-valid starter strategy. It ignores coverage structure and
simply returns a seeded random subset of pool indices without replacement.
"""


def _get_pool(context):
    return np.asarray(context.get("pool_unit", context["pool"]), dtype=np.float64)


def _get_seed(context) -> int:
    metadata = context.get("metadata", {})
    base_seed = int(metadata.get("seed", 123))
    return base_seed + 1009


def entrypoint(context):
    pool = _get_pool(context)
    m = int(pool.shape[0])
    k = int(context["target_k"])

    if k >= m:
        return np.arange(m, dtype=np.int64)

    rng = np.random.default_rng(_get_seed(context))
    choice = rng.choice(m, size=k, replace=False)
    return np.asarray(choice, dtype=np.int64)
