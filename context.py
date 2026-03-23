from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

"""
Runtime context builder for the E8 direction-codebook problem.

This file is used only when the GigaEvo problem is contextual.
GigaEvo will call build_context() once at startup and pass the returned dict to:
    - entrypoint(context)
    - validate(context, program_output)

The goal here is to keep the E8 pool fixed and let evolution optimize only the
subset-selection algorithm.
"""

DIM = 8
DEFAULT_TARGET_K = 2 ** 16
DEFAULT_SEARCH_NUM_SAMPLES = 8192
DEFAULT_FINAL_NUM_SAMPLES = 32768
DEFAULT_SPHERE_RADIUS = 32.0
DEFAULT_SEED = 123
DEFAULT_PAIRWISE_PENALTY_WEIGHT = 0.25
DEFAULT_MIN_POOL_SIZE = 1024
DEFAULT_PAIRWISE_BLOCK_SIZE = 1024
DEFAULT_CONTEXT_MODE = "search"
CACHE_DIRNAME = ".cache"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {name}: {value!r}")


def _resolve_mode() -> str:
    raw = os.getenv("E8_CONTEXT_MODE") or os.getenv("E8_EVAL_MODE") or DEFAULT_CONTEXT_MODE
    mode = raw.strip().lower()
    if mode not in {"search", "final"}:
        raise ValueError(f"Invalid context mode: {raw!r}. Expected 'search' or 'final'.")
    return mode


def _resolve_num_samples(mode: str) -> tuple[int, str]:
    explicit = os.getenv("E8_NUM_SAMPLES")
    if explicit is not None:
        return int(explicit), "E8_NUM_SAMPLES"

    if mode == "search":
        return _env_int("E8_SEARCH_NUM_SAMPLES", DEFAULT_SEARCH_NUM_SAMPLES), "search-default"
    return _env_int("E8_FINAL_NUM_SAMPLES", DEFAULT_FINAL_NUM_SAMPLES), "final-default"


def _resolve_min_pool_size(target_k: int) -> int:
    return _env_int("E8_MIN_POOL_SIZE", max(DEFAULT_MIN_POOL_SIZE, target_k))


def _suggest_coverage_batch_size(num_samples: int) -> int:
    # This batch size controls how many centers validate.py processes per matrix multiply.
    # Larger sample sets already increase the temporary GEMM size, so use a smaller center batch.
    if num_samples >= 32768:
        return 256
    if num_samples >= 16384:
        return 512
    return 1024


def _validate_shells(shells: Sequence[int]) -> None:
    if not shells:
        raise ValueError("At least one E8 shell must be specified")
    if any(s <= 0 or s % 2 != 0 for s in shells):
        raise ValueError("E8 shell squared norms must be positive even integers")


@lru_cache(maxsize=None)
def _possible_values(parity: int, remaining_sq: int) -> tuple[int, ...]:
    """All integers of the chosen parity whose square fits in remaining_sq."""
    limit = math.isqrt(remaining_sq)
    values: list[int] = []
    start = parity % 2
    for abs_v in range(start, limit + 1, 2):
        sq = abs_v * abs_v
        if sq > remaining_sq:
            break
        if abs_v == 0:
            values.append(0)
        else:
            values.extend((-abs_v, abs_v))
    values.sort(key=lambda x: (x * x, x))
    return tuple(values)


def _enumerate_scaled_shell_vectors(norm_sq: int) -> Iterator[tuple[int, ...]]:
    """
    Enumerate all non-zero E8 vectors x with ||x||^2 = norm_sq.

    We use the scaled integer representation y = 2x.
    Then E8 can be written as:
        y in Z^8,
        all coordinates have the same parity,
        sum(y_i) == 0 mod 4.
    The shell condition becomes ||y||^2 = 4 * norm_sq.
    """
    target_scaled_sq = 4 * norm_sq
    for parity in (0, 1):
        current = [0] * DIM

        def backtrack(pos: int, remaining_sq: int) -> Iterator[tuple[int, ...]]:
            if pos == DIM:
                if remaining_sq == 0 and (sum(current) % 4 == 0):
                    yield tuple(current)
                return

            for value in _possible_values(parity, remaining_sq):
                current[pos] = value
                yield from backtrack(pos + 1, remaining_sq - value * value)
            current[pos] = 0

        yield from backtrack(0, target_scaled_sq)


def _primitive_key_from_scaled(vector: Sequence[int], merge_antipodal: bool) -> tuple[int, ...]:
    """
    Canonical direction key for deduplication across shells.

    Distinct lattice vectors on the same ray should map to the same key.
    By default, v and -v remain different directions.
    If merge_antipodal=True, v and -v are identified.
    """
    arr = np.asarray(vector, dtype=np.int64)
    gcd = int(np.gcd.reduce(np.abs(arr)))
    if gcd == 0:
        raise ValueError("Zero vector is not a valid E8 direction")
    primitive = (arr // gcd).astype(np.int64)
    if merge_antipodal:
        nz = np.flatnonzero(primitive)
        if nz.size and primitive[nz[0]] < 0:
            primitive = -primitive
    return tuple(int(x) for x in primitive.tolist())


def _scaled_to_unit(vector: Sequence[int]) -> np.ndarray:
    x = 0.5 * np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm == 0.0:
        raise ValueError("Zero vector is not a valid E8 direction")
    return x / norm


def _default_base_dir() -> Path:
    file_value = globals().get("__file__")
    if file_value:
        try:
            return Path(file_value).resolve().parent
        except Exception:
            pass
    return Path.cwd()


def _resolve_cache_dir() -> Path | None:
    if _env_bool("E8_DISABLE_CACHE", False):
        return None

    explicit = os.getenv("E8_CACHE_DIR")
    cache_dir = Path(explicit).expanduser() if explicit else (_default_base_dir() / CACHE_DIRNAME)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return cache_dir


def _cache_path(shells: Sequence[int], merge_antipodal: bool) -> Path | None:
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return None
    shell_part = "-".join(str(s) for s in shells)
    antipodal_part = "merged" if merge_antipodal else "signed"
    return cache_dir / f"e8_pool_shells_{shell_part}_{antipodal_part}.npz"


def _load_cached_pool(shells: Sequence[int], merge_antipodal: bool) -> np.ndarray | None:
    path = _cache_path(shells, merge_antipodal)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            pool = np.asarray(data["pool"], dtype=np.float64)
        if pool.ndim == 2 and pool.shape[1] == DIM and pool.shape[0] > 0:
            return pool
    except Exception:
        return None
    return None


def _save_cached_pool(shells: Sequence[int], merge_antipodal: bool, pool: np.ndarray) -> None:
    path = _cache_path(shells, merge_antipodal)
    if path is None:
        return
    try:
        np.savez_compressed(path, pool=pool)
    except Exception:
        return


def build_e8_direction_pool(
    shell_norm_sq_values: Sequence[int],
    *,
    merge_antipodal: bool = False,
) -> np.ndarray:
    """
    Build a pool of unique unit directions induced by the selected E8 shells.

    Args:
        shell_norm_sq_values: even squared norms of E8 lattice shells to include.
        merge_antipodal: whether to identify v and -v as the same direction.

    Returns:
        pool: ndarray of shape (M, 8), each row a unit direction.
    """
    shells = tuple(sorted({int(s) for s in shell_norm_sq_values}))
    _validate_shells(shells)

    cached = _load_cached_pool(shells, merge_antipodal)
    if cached is not None:
        return cached

    seen: set[tuple[int, ...]] = set()
    directions: list[np.ndarray] = []

    for shell_sq in shells:
        for scaled_vec in _enumerate_scaled_shell_vectors(shell_sq):
            key = _primitive_key_from_scaled(scaled_vec, merge_antipodal=merge_antipodal)
            if key in seen:
                continue
            seen.add(key)
            directions.append(_scaled_to_unit(scaled_vec))

    if not directions:
        raise ValueError("No E8 directions were generated; check shell configuration")

    pool = np.stack(directions, axis=0).astype(np.float64, copy=False)
    _save_cached_pool(shells, merge_antipodal, pool)
    return pool


def _select_shells_for_min_pool(min_pool_size: int, merge_antipodal: bool) -> tuple[int, ...]:
    """
    Add shells in ascending order until the direction pool is large enough.

    This gives a finite approximation to the infinite set of E8 directions while
    keeping the construction deterministic and easy to reproduce.
    """
    if min_pool_size <= 0:
        raise ValueError("min_pool_size must be positive")

    shells: list[int] = []
    shell_sq = 2
    max_shell_sq = _env_int("E8_AUTO_MAX_SHELL_NORM_SQ", 32)

    while shell_sq <= max_shell_sq:
        candidate_shells = tuple(shells + [shell_sq])
        pool = build_e8_direction_pool(candidate_shells, merge_antipodal=merge_antipodal)
        shells.append(shell_sq)
        if len(pool) >= min_pool_size:
            return tuple(shells)
        shell_sq += 2

    if not shells:
        raise ValueError("Automatic shell selection failed to generate any shells")
    return tuple(shells)


def _resolve_shells(target_k: int, merge_antipodal: bool) -> tuple[int, ...]:
    explicit = os.getenv("E8_SHELL_NORM_SQ")
    if explicit:
        shells = tuple(sorted({int(x.strip()) for x in explicit.split(",") if x.strip()}))
        _validate_shells(shells)
        return shells

    max_shell = os.getenv("E8_MAX_SHELL_NORM_SQ")
    if max_shell is not None:
        max_sq = int(max_shell)
        if max_sq < 2 or max_sq % 2 != 0:
            raise ValueError("E8_MAX_SHELL_NORM_SQ must be an even integer >= 2")
        shells = tuple(range(2, max_sq + 1, 2))
        _validate_shells(shells)
        return shells

    min_pool_size = _resolve_min_pool_size(target_k)
    return _select_shells_for_min_pool(
        min_pool_size=min_pool_size,
        merge_antipodal=merge_antipodal,
    )


def random_unit_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    zero_mask = norms[:, 0] == 0.0
    while np.any(zero_mask):
        x[zero_mask] = rng.normal(size=(int(np.sum(zero_mask)), dim))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        zero_mask = norms[:, 0] == 0.0
    x /= norms
    return x.astype(np.float64, copy=False)


def build_context() -> dict:
    """
    Build the runtime context consumed by initial programs and validate.py.

    Supported environment overrides:
        E8_CONTEXT_MODE              search / final
        E8_EVAL_MODE                 alias for E8_CONTEXT_MODE
        E8_TARGET_K
        E8_NUM_SAMPLES               explicit override for either mode
        E8_SEARCH_NUM_SAMPLES        mode-specific override (default 8192)
        E8_FINAL_NUM_SAMPLES         mode-specific override (default 32768)
        E8_RANDOM_SEED
        E8_SPHERE_RADIUS
        E8_PAIRWISE_PENALTY_WEIGHT
        E8_COVERAGE_BATCH_SIZE
        E8_PAIRWISE_BLOCK_SIZE
        E8_SHELL_NORM_SQ             e.g. "2,4,6,8"
        E8_MAX_SHELL_NORM_SQ         e.g. "8"
        E8_MIN_POOL_SIZE             e.g. "20000"
        E8_AUTO_MAX_SHELL_NORM_SQ
        E8_MERGE_ANTIPODAL           0/1
        E8_CACHE_DIR                 explicit cache directory
        E8_DISABLE_CACHE             0/1

    Geometry convention:
        - context["pool"] and context["samples"] lie on a sphere of radius 32 by default
        - context["pool_unit"] and context["samples_unit"] keep the corresponding unit vectors

    For target_k = 2**16, the intended defaults are:
        - search: 8192 probe samples
        - final : 32768 probe samples
    """
    mode = _resolve_mode()
    target_k = _env_int("E8_TARGET_K", DEFAULT_TARGET_K)
    num_samples, num_samples_source = _resolve_num_samples(mode)
    seed = _env_int("E8_RANDOM_SEED", DEFAULT_SEED)
    sphere_radius = _env_float("E8_SPHERE_RADIUS", DEFAULT_SPHERE_RADIUS)
    pairwise_penalty_weight = _env_float(
        "E8_PAIRWISE_PENALTY_WEIGHT",
        DEFAULT_PAIRWISE_PENALTY_WEIGHT,
    )
    coverage_batch_size = _env_int(
        "E8_COVERAGE_BATCH_SIZE",
        _suggest_coverage_batch_size(num_samples),
    )
    pairwise_block_size = _env_int("E8_PAIRWISE_BLOCK_SIZE", DEFAULT_PAIRWISE_BLOCK_SIZE)
    merge_antipodal = _env_bool("E8_MERGE_ANTIPODAL", False)
    shells = _resolve_shells(target_k=target_k, merge_antipodal=merge_antipodal)

    if target_k <= 0:
        raise ValueError("E8_TARGET_K must be positive")
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive")
    if sphere_radius <= 0.0:
        raise ValueError("E8_SPHERE_RADIUS must be positive")
    if coverage_batch_size <= 0 or pairwise_block_size <= 0:
        raise ValueError("Batch sizes must be positive")

    pool_unit = build_e8_direction_pool(shell_norm_sq_values=shells, merge_antipodal=merge_antipodal)
    if len(pool_unit) < target_k:
        raise ValueError(
            f"Pool size {len(pool_unit)} is smaller than target_k={target_k}. "
            "Increase E8_SHELL_NORM_SQ / E8_MAX_SHELL_NORM_SQ / E8_MIN_POOL_SIZE or lower E8_TARGET_K."
        )

    # Use an offset so the sample set stays stable when only target_k changes.
    samples_unit = random_unit_vectors(n=num_samples, dim=DIM, seed=seed + 1)
    pool = (sphere_radius * pool_unit).astype(np.float64, copy=False)
    samples = (sphere_radius * samples_unit).astype(np.float64, copy=False)

    return {
        "pool": pool,
        "pool_unit": pool_unit,
        "samples": samples,
        "samples_unit": samples_unit,
        "target_k": int(target_k),
        "pairwise_penalty_weight": float(pairwise_penalty_weight),
        "coverage_batch_size": int(coverage_batch_size),
        "pairwise_block_size": int(pairwise_block_size),
        "metadata": {
            "mode": mode,
            "dimension": DIM,
            "shell_norm_sq_values": tuple(int(s) for s in shells),
            "pool_size": int(len(pool_unit)),
            "sphere_radius": float(sphere_radius),
            "min_pool_size_request": int(_resolve_min_pool_size(target_k)),
            "num_samples": int(num_samples),
            "num_samples_source": str(num_samples_source),
            "seed": int(seed),
            "merge_antipodal": bool(merge_antipodal),
            "coverage_batch_size": int(coverage_batch_size),
            "pairwise_block_size": int(pairwise_block_size),
            "cache_dir": str(_resolve_cache_dir()) if _resolve_cache_dir() is not None else None,
        },
    }


if __name__ == "__main__":
    ctx = build_context()
    print(
        {
            "pool_shape": tuple(ctx["pool"].shape),
            "samples_shape": tuple(ctx["samples"].shape),
            **ctx["metadata"],
        }
    )
