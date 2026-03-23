from __future__ import annotations

import numpy as np


"""
Fast deterministic spread selection using random projections.

Idea:
1. Build several deterministic random projection directions.
2. Sort the pool along each projection.
3. Pick evenly spaced indices from every sorted order.
4. Deduplicate and fill any remaining slots from a deterministic permutation.

This scales much better than full greedy maximin when target_k is large.
"""


def _get_pool(context):
    return np.asarray(context.get("pool_unit", context["pool"]), dtype=np.float64)


def _get_seed(context) -> int:
    metadata = context.get("metadata", {})
    base_seed = int(metadata.get("seed", 123))
    return base_seed + 2027


def _deterministic_permutation(pool: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a deterministic permutation using projection scores instead of rng.permutation."""
    anchor_1 = rng.normal(size=pool.shape[1])
    anchor_2 = rng.normal(size=pool.shape[1])
    anchor_1 /= np.linalg.norm(anchor_1)
    anchor_2 /= np.linalg.norm(anchor_2)
    scores = pool @ anchor_1
    tiebreak = pool @ anchor_2
    order = np.lexsort((tiebreak, scores))
    return np.asarray(order, dtype=np.int64)


def _projection_spread(pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    m, d = pool.shape
    if k >= m:
        return np.arange(m, dtype=np.int64)

    n_proj = int(np.clip(np.ceil(np.log2(max(k, 2))), 4, 24))
    projections = rng.normal(size=(n_proj, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    target_unique = min(m, max(k, int(1.5 * k), 512))
    per_proj = max(2, int(np.ceil(target_unique / n_proj)))

    seen = np.zeros(m, dtype=bool)
    picked: list[int] = []

    for p_idx, direction in enumerate(projections):
        order = np.argsort(pool @ direction)

        if per_proj >= m:
            candidates = order
        else:
            positions = np.linspace(0, m - 1, num=per_proj, dtype=np.int64)
            if per_proj > 1:
                step = max(1, m // (per_proj + 1))
                offset = (p_idx * step) // max(1, n_proj - 1)
                positions = np.minimum(m - 1, positions + offset)
            candidates = order[positions]

        for idx in candidates:
            idx = int(idx)
            if not seen[idx]:
                seen[idx] = True
                picked.append(idx)
                if len(picked) >= target_unique:
                    break
        if len(picked) >= target_unique:
            break

    if len(picked) < k:
        perm = _deterministic_permutation(pool, rng)
        for idx in perm:
            idx = int(idx)
            if not seen[idx]:
                seen[idx] = True
                picked.append(idx)
                if len(picked) >= k:
                    break

    return np.asarray(picked[:k], dtype=np.int64)


def entrypoint(context):
    pool = _get_pool(context)
    k = int(context["target_k"])
    rng = np.random.default_rng(_get_seed(context))
    return _projection_spread(pool=pool, k=k, rng=rng)
