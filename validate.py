from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback when SciPy is unavailable
    cKDTree = None
    _HAS_SCIPY = False

# Set to True during local debugging if you want validator errors to surface
# instead of being converted into an invalid result.
DEBUG_RAISE = False

# Fallback values returned for invalid outputs.
# These are deliberately pessimistic within the natural ranges of the metrics.
_INVALID_RESULT = {
    "fitness": -1.25,
    "mean_cover": -1.0,
    "worst_cover": -1.0,
    "max_pair_cos": 1.0,
    "is_valid": 0.0,
}


def _invalid_result() -> dict[str, float]:
    """Return a fresh invalid-result payload."""
    return dict(_INVALID_RESULT)


def _get_context_array(context: dict[str, Any], primary_key: str, fallback_key: str) -> Any:
    """Prefer pre-normalized arrays when present, otherwise use the base key."""
    if primary_key in context:
        return context[primary_key]
    return context[fallback_key]


def _normalize_rows(x: np.ndarray) -> np.ndarray | None:
    """Return a row-wise L2-normalized copy, or None when a zero/NaN row exists."""
    if x.ndim != 2:
        return None
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return None
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        return None
    return x / norms


def _parse_context(context: Any) -> tuple[np.ndarray, np.ndarray, int, float, int, int, str, int] | None:
    """
    Extract and validate the runtime context.

    Expected keys:
      - pool or pool_unit:        (M, 8)
      - samples or samples_unit:  (N, 8)
      - target_k: int

    Optional keys:
      - pairwise_penalty_weight: float, default 0.25
      - coverage_batch_size: int, default 512
      - pairwise_block_size: int, default 1024
      - validator_backend: "auto" | "kdtree" | "gemm"
      - distance_query_workers: int, default -1

    Notes:
      - When SciPy is available and backend resolves to "kdtree", both coverage
        and max_pair_cos are computed exactly via nearest-neighbor queries in
        Euclidean distance. For unit vectors this is equivalent to cosine search
        because ||u - v||^2 = 2 - 2<u, v>.
      - The GEMM backend is kept as a dependency-free fallback.
    """
    if not isinstance(context, dict):
        return None

    try:
        pool = np.asarray(_get_context_array(context, "pool_unit", "pool"))
        samples = np.asarray(_get_context_array(context, "samples_unit", "samples"))
        target_k = int(context["target_k"])
        penalty_weight = float(context.get("pairwise_penalty_weight", 0.25))
        coverage_batch_size = int(context.get("coverage_batch_size", 512))
        pairwise_block_size = int(context.get("pairwise_block_size", 1024))
        backend = str(context.get("validator_backend", "auto")).strip().lower()
        distance_query_workers = int(context.get("distance_query_workers", -1))
    except Exception:
        return None

    if pool.ndim != 2 or samples.ndim != 2:
        return None
    if pool.shape[1] != 8 or samples.shape[1] != 8:
        return None
    if pool.shape[0] == 0 or samples.shape[0] == 0:
        return None
    if target_k <= 0 or target_k > pool.shape[0]:
        return None
    if coverage_batch_size <= 0 or pairwise_block_size <= 0:
        return None
    if not np.issubdtype(pool.dtype, np.number):
        return None
    if not np.issubdtype(samples.dtype, np.number):
        return None
    if not np.all(np.isfinite(pool)) or not np.all(np.isfinite(samples)):
        return None
    if not np.isfinite(penalty_weight):
        return None
    if backend not in {"auto", "kdtree", "gemm"}:
        return None

    # Normalize once here so both backends operate on unit vectors. This makes
    # the KD-tree path exact for cosine objectives and keeps behavior robust even
    # if the context arrays are only approximately normalized.
    pool_unit = _normalize_rows(pool)
    samples_unit = _normalize_rows(samples)
    if pool_unit is None or samples_unit is None:
        return None

    if backend == "auto":
        resolved_backend = "kdtree" if _HAS_SCIPY else "gemm"
    elif backend == "kdtree" and not _HAS_SCIPY:
        resolved_backend = "gemm"
    else:
        resolved_backend = backend

    return (
        pool_unit,
        samples_unit,
        target_k,
        penalty_weight,
        coverage_batch_size,
        pairwise_block_size,
        resolved_backend,
        distance_query_workers,
    )


def _parse_indices(indices: Any) -> np.ndarray | None:
    """
    Convert the program output into a flat int64 index array.

    Accepted inputs: list/tuple/numpy array of integers, or floats that are exactly
    integral (e.g. 3.0). Strings, scalars, booleans, NaNs, and non-integral floats
    are rejected.
    """
    if isinstance(indices, (str, bytes)):
        return None

    try:
        arr = np.asarray(indices)
    except Exception:
        return None

    if arr.ndim == 0:
        return None

    arr = np.ravel(arr)
    if arr.size == 0:
        return None

    if np.issubdtype(arr.dtype, np.bool_):
        return None

    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)

    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            return None
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded, rtol=0.0, atol=1e-9):
            return None
        return rounded.astype(np.int64)

    return None


def _coverage_stats_gemm(samples: np.ndarray, centers: np.ndarray, batch_size: int) -> tuple[float, float]:
    """
    Compute coverage metrics without materializing the full (N, K) similarity matrix.

    mean_cover  = mean_x max_j <x, c_j>
    worst_cover = min_x  max_j <x, c_j>
    """
    n_samples = samples.shape[0]
    best_sim = np.full(n_samples, -np.inf, dtype=np.float64)

    for start in range(0, centers.shape[0], batch_size):
        stop = min(start + batch_size, centers.shape[0])
        block = centers[start:stop]
        sims = samples @ block.T
        block_best = np.max(sims, axis=1)
        np.maximum(best_sim, block_best, out=best_sim)

    return float(np.mean(best_sim)), float(np.min(best_sim))


def _max_pair_cos_gemm(centers: np.ndarray, block_size: int) -> float:
    """
    Compute the maximum pairwise cosine among selected centers using blocked GEMMs.

    This is exact, but memory-safe: it never materializes the full (K, K) Gram matrix.
    """
    k = centers.shape[0]
    if k < 2:
        return 0.0

    max_cos = -1.0

    for i in range(0, k, block_size):
        i_stop = min(i + block_size, k)
        block_i = centers[i:i_stop]

        gram_ii = block_i @ block_i.T
        if gram_ii.shape[0] > 1:
            tri_rows, tri_cols = np.triu_indices(gram_ii.shape[0], k=1)
            local_max = float(np.max(gram_ii[tri_rows, tri_cols]))
            if local_max > max_cos:
                max_cos = local_max

        for j in range(i_stop, k, block_size):
            j_stop = min(j + block_size, k)
            block_j = centers[j:j_stop]
            gram_ij = block_i @ block_j.T
            local_max = float(np.max(gram_ij))
            if local_max > max_cos:
                max_cos = local_max

    return max_cos if max_cos > -1.0 else 0.0


def _tree_query(tree: Any, points: np.ndarray, k: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for cKDTree.query across SciPy versions."""
    try:
        return tree.query(points, k=k, workers=workers)
    except TypeError:
        try:
            return tree.query(points, k=k, n_jobs=workers)
        except TypeError:
            return tree.query(points, k=k)


def _coverage_and_pairwise_kdtree(
    samples: np.ndarray,
    centers: np.ndarray,
    workers: int,
) -> tuple[float, float, float]:
    """
    Exact cosine metrics via Euclidean nearest-neighbor search on unit vectors.

    For unit vectors u, v:
        ||u - v||^2 = 2 - 2<u, v>
        <u, v> = 1 - 0.5 * ||u - v||^2

    Therefore:
      - nearest center in Euclidean distance == best center in cosine similarity;
      - nearest other center gives the pair with maximum cosine.
    """
    tree = cKDTree(centers)

    sample_dist, _ = _tree_query(tree, samples, k=1, workers=workers)
    sample_dist = np.asarray(sample_dist, dtype=np.float64)
    best_sim = 1.0 - 0.5 * np.square(sample_dist)
    mean_cover = float(np.mean(best_sim))
    worst_cover = float(np.min(best_sim))

    if centers.shape[0] < 2:
        max_pair_cos = 0.0
    else:
        center_dist, _ = _tree_query(tree, centers, k=2, workers=workers)
        center_dist = np.asarray(center_dist, dtype=np.float64)
        # center_dist[:, 0] is distance to self == 0. The second column is the
        # nearest distinct center because duplicates were rejected earlier.
        min_nn_dist = float(np.min(center_dist[:, 1]))
        max_pair_cos = float(1.0 - 0.5 * (min_nn_dist * min_nn_dist))
        # Numerical guard against tiny overshoots above 1 caused by roundoff.
        max_pair_cos = min(1.0, max(-1.0, max_pair_cos))

    return mean_cover, worst_cover, max_pair_cos


def validate(context: dict[str, Any], indices: Any) -> dict[str, float]:
    """
    Validate a candidate E8-based directional codebook.

    The program output must be a 1D array-like object of unique indices into
    context["pool"] (or context["pool_unit"] when pre-normalized data is provided).

    Returned metrics:
      - fitness      : mean_cover - lambda * max_pair_cos
      - mean_cover   : average nearest-center cosine over sphere samples
      - worst_cover  : worst nearest-center cosine over sphere samples
      - max_pair_cos : largest pairwise cosine among selected centers
      - is_valid     : 1.0 for valid outputs, 0.0 otherwise
    """
    try:
        parsed = _parse_context(context)
        if parsed is None:
            return _invalid_result()

        (
            pool,
            samples,
            target_k,
            penalty_weight,
            coverage_batch_size,
            pairwise_block_size,
            backend,
            distance_query_workers,
        ) = parsed

        idx = _parse_indices(indices)
        if idx is None:
            return _invalid_result()
        if idx.size != target_k:
            return _invalid_result()
        if np.any(idx < 0) or np.any(idx >= pool.shape[0]):
            return _invalid_result()
        if np.unique(idx).size != idx.size:
            return _invalid_result()

        centers = pool[idx]

        if backend == "kdtree":
            mean_cover, worst_cover, max_pair_cos = _coverage_and_pairwise_kdtree(
                samples=samples,
                centers=centers,
                workers=distance_query_workers,
            )
        else:
            mean_cover, worst_cover = _coverage_stats_gemm(
                samples=samples,
                centers=centers,
                batch_size=coverage_batch_size,
            )
            max_pair_cos = _max_pair_cos_gemm(
                centers=centers,
                block_size=pairwise_block_size,
            )

        fitness = float(mean_cover - penalty_weight * max_pair_cos)

        return {
            "fitness": fitness,
            "mean_cover": float(mean_cover),
            "worst_cover": float(worst_cover),
            "max_pair_cos": float(max_pair_cos),
            "is_valid": 1.0,
        }
    except Exception:
        if DEBUG_RAISE:
            raise
        return _invalid_result()
