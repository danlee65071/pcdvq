from __future__ import annotations

from collections import defaultdict

import numpy as np


def _get_pool(context: dict) -> np.ndarray:
    if "pool_unit" in context:
        return np.asarray(context["pool_unit"], dtype=np.float64)
    return np.asarray(context["pool"], dtype=np.float64)


def _get_seed(context: dict) -> int:
    metadata = context.get("metadata")
    if isinstance(metadata, dict) and "seed" in metadata:
        return int(metadata["seed"])
    return int(context.get("seed", 0))


def _projection_round_robin(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    m, dim = pool.shape
    num_planes = int(np.clip(round(np.log2(max(k, 2))) + 2, 6, 14))
    rng = np.random.default_rng(seed + 17)
    planes = rng.normal(size=(num_planes, dim))
    planes /= np.linalg.norm(planes, axis=1, keepdims=True)
    projections = pool @ planes.T

    bits = (projections >= 0.0).astype(np.uint64, copy=False)
    weights = (np.uint64(1) << np.arange(num_planes, dtype=np.uint64))
    bucket_id = bits @ weights
    local_score = projections @ np.linspace(1.0, 2.0, num_planes, dtype=np.float64)

    grouped: dict[int, list[int]] = defaultdict(list)
    order = np.lexsort((-local_score, bucket_id))
    for idx in order:
        grouped[int(bucket_id[idx])].append(int(idx))

    bucket_order = sorted(grouped.keys(), key=lambda b: (-len(grouped[b]), b))
    positions = {b: 0 for b in bucket_order}
    selected: list[int] = []

    while len(selected) < k:
        progressed = False
        for b in bucket_order:
            pos = positions[b]
            bucket = grouped[b]
            if pos >= len(bucket):
                continue
            selected.append(bucket[pos])
            positions[b] = pos + 1
            progressed = True
            if len(selected) >= k:
                break
        if not progressed:
            break

    if len(selected) < k:
        chosen_mask = np.zeros(m, dtype=bool)
        chosen_mask[np.asarray(selected, dtype=np.int64)] = True
        remaining = np.flatnonzero(~chosen_mask)
        fill_order = remaining[np.argsort(-local_score[remaining], kind="mergesort")]
        selected.extend(int(i) for i in fill_order[: k - len(selected)])

    return np.asarray(selected[:k], dtype=np.int64)


def _first_index(pool: np.ndarray, seed: int) -> int:
    rng = np.random.default_rng(seed + 1)
    anchor = rng.normal(size=(pool.shape[1],))
    anchor_norm = np.linalg.norm(anchor)
    if anchor_norm == 0.0:
        anchor[0] = 1.0
        anchor_norm = 1.0
    anchor /= anchor_norm
    return int(np.argmax(pool @ anchor))


def _exact_greedy(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    m = pool.shape[0]
    selected = np.empty(k, dtype=np.int64)
    chosen_mask = np.zeros(m, dtype=bool)

    first = _first_index(pool, seed)
    selected[0] = first
    chosen_mask[first] = True

    best_sim = pool @ pool[first]
    best_sim[first] = np.inf

    for t in range(1, k):
        nxt = int(np.argmin(best_sim))
        selected[t] = nxt
        chosen_mask[nxt] = True

        sims = pool @ pool[nxt]
        np.maximum(best_sim, sims, out=best_sim)
        best_sim[chosen_mask] = np.inf

    return selected


def entrypoint(context: dict):
    """
    Strong baseline for small and medium problems.

    Exact mode: greedy farthest-point sampling under cosine similarity.
    Fallback mode: when the exact O(M * K) loop would be too expensive, switch to
    a projection-stratified round-robin subset that remains deterministic and fast.
    """
    pool = _get_pool(context)
    k = int(context["target_k"])
    m = int(pool.shape[0])
    if k <= 0:
        raise ValueError("target_k must be positive")
    if k > m:
        raise ValueError(f"target_k={k} exceeds pool size {m}")

    seed = _get_seed(context)
    exact_budget = int(context.get("greedy_exact_budget", 25_000_000))

    if m * k > exact_budget:
        return _projection_round_robin(pool=pool, k=k, seed=seed)
    return _exact_greedy(pool=pool, k=k, seed=seed)
