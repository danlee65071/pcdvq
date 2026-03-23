from __future__ import annotations

import numpy as np


"""
Hybrid starter:
1. Build a projection-spread candidate subset from the full pool.
2. Run exact greedy maximin inside that smaller candidate set.

This often gives a stronger starting point than pure random or pure projection
selection while remaining much cheaper than full greedy over the whole pool.
"""


CANDIDATE_CAP = 16384
CANDIDATE_FACTOR = 4
GREEDY_OPS_BUDGET = 2.0e8


def _get_pool(context):
    return np.asarray(context.get("pool_unit", context["pool"]), dtype=np.float64)


def _get_seed(context) -> int:
    metadata = context.get("metadata", {})
    base_seed = int(metadata.get("seed", 123))
    return base_seed + 4093


def _projection_candidates(pool: np.ndarray, candidate_k: int, rng: np.random.Generator) -> np.ndarray:
    m, d = pool.shape
    if candidate_k >= m:
        return np.arange(m, dtype=np.int64)

    n_proj = int(np.clip(np.ceil(np.log2(max(candidate_k, 2))), 4, 20))
    projections = rng.normal(size=(n_proj, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    seen = np.zeros(m, dtype=bool)
    picked: list[int] = []
    per_proj = max(2, int(np.ceil(candidate_k / n_proj)))

    for p_idx, direction in enumerate(projections):
        order = np.argsort(pool @ direction)
        positions = np.linspace(0, m - 1, per_proj, dtype=np.int64)
        if per_proj > 1:
            positions = np.minimum(m - 1, positions + p_idx)
        for idx in order[positions]:
            idx = int(idx)
            if not seen[idx]:
                seen[idx] = True
                picked.append(idx)
                if len(picked) >= candidate_k:
                    return np.asarray(picked, dtype=np.int64)

    if len(picked) < candidate_k:
        anchor = projections[0]
        tail = np.argsort(pool @ anchor)
        for idx in tail:
            idx = int(idx)
            if not seen[idx]:
                seen[idx] = True
                picked.append(idx)
                if len(picked) >= candidate_k:
                    break

    return np.asarray(picked[:candidate_k], dtype=np.int64)


def _greedy_within_candidates(candidate_pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    c = candidate_pool.shape[0]
    if k >= c:
        return np.arange(c, dtype=np.int64)

    anchor = rng.normal(size=candidate_pool.shape[1])
    anchor /= np.linalg.norm(anchor)
    first = int(np.argmax(candidate_pool @ anchor))

    selected = np.empty(k, dtype=np.int64)
    selected[0] = first
    chosen = np.zeros(c, dtype=bool)
    chosen[first] = True

    best_sim = candidate_pool @ candidate_pool[first]
    best_sim[first] = np.inf

    for t in range(1, k):
        next_idx = int(np.argmin(best_sim))
        selected[t] = next_idx
        chosen[next_idx] = True
        sims = candidate_pool @ candidate_pool[next_idx]
        np.maximum(best_sim, sims, out=best_sim)
        best_sim[chosen] = np.inf

    return selected


def entrypoint(context):
    pool = _get_pool(context)
    m = int(pool.shape[0])
    k = int(context["target_k"])
    rng = np.random.default_rng(_get_seed(context))

    if k >= m:
        return np.arange(m, dtype=np.int64)

    candidate_k = int(min(m, max(CANDIDATE_FACTOR * k, 2048, k), CANDIDATE_CAP))

    if k >= candidate_k or float(candidate_k) * float(k) > GREEDY_OPS_BUDGET:
        return _projection_candidates(pool=pool, candidate_k=k, rng=rng)

    candidate_indices = _projection_candidates(pool=pool, candidate_k=candidate_k, rng=rng)
    candidate_pool = pool[candidate_indices]
    local_choice = _greedy_within_candidates(candidate_pool=candidate_pool, k=k, rng=rng)
    return np.asarray(candidate_indices[local_choice], dtype=np.int64)
