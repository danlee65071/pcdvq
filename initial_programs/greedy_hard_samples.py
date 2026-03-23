import numpy as np


def _subsample_rows(x, limit):
    n = x.shape[0]
    if n <= limit:
        return x
    idx = np.linspace(0, n - 1, num=limit, dtype=np.int64)
    return x[idx]


def entrypoint(context):
    """
    Coverage-aware greedy construction.

    Uses a moderate subset of sphere samples, tracks current coverage, focuses on
    the hardest-covered samples, and adds a separation penalty through the
    running maximum pairwise cosine to the selected set.
    """
    pool = np.asarray(context.get("pool_unit", context["pool"]), dtype=np.float64)
    samples = np.asarray(context.get("samples_unit", context["samples"]), dtype=np.float64)
    k = int(context["target_k"])
    penalty = float(context.get("pairwise_penalty_weight", 0.25))

    m = int(pool.shape[0])
    if k >= m:
        return np.arange(m, dtype=np.int64)

    eval_samples = _subsample_rows(samples, limit=min(1024, max(256, 4 * k)))
    q = eval_samples.shape[0]
    hard_count = min(64, q)

    selected = np.empty(k, dtype=np.int64)
    chosen_mask = np.zeros(m, dtype=bool)
    running_max_sim = np.full(m, -1.0, dtype=np.float64)
    coverage = np.full(q, -1.0, dtype=np.float64)

    initial_scores = eval_samples @ pool.T
    first = int(np.argmax(initial_scores.mean(axis=0)))
    selected[0] = first
    chosen_mask[first] = True
    running_max_sim = np.maximum(running_max_sim, pool @ pool[first])
    running_max_sim[first] = 1.0
    coverage = np.maximum(coverage, eval_samples @ pool[first])

    for t in range(1, k):
        hard_idx = np.argpartition(coverage, hard_count - 1)[:hard_count]
        hard = eval_samples[hard_idx]
        hard_scores = hard @ pool.T

        hard_mean = hard_scores.mean(axis=0)
        hard_min = hard_scores.min(axis=0)
        separation_penalty = penalty * np.clip(running_max_sim, 0.0, None)
        score = 0.7 * hard_mean + 0.3 * hard_min - separation_penalty
        score[chosen_mask] = -np.inf

        next_idx = int(np.argmax(score))
        selected[t] = next_idx
        chosen_mask[next_idx] = True

        np.maximum(running_max_sim, pool @ pool[next_idx], out=running_max_sim)
        running_max_sim[chosen_mask] = 1.0
        np.maximum(coverage, eval_samples @ pool[next_idx], out=coverage)

    return selected
