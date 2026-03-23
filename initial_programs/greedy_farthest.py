import numpy as np


def entrypoint(context):
    """
    Greedy farthest-point sampling in cosine geometry.

    This mirrors the E8 direction-selection intuition from the paper: after the
    first center, repeatedly add the pool element whose maximum cosine to the
    already-selected set is minimal.
    """
    pool = np.asarray(context.get("pool_unit", context["pool"]), dtype=np.float64)
    samples = np.asarray(context.get("samples_unit", context["samples"]), dtype=np.float64)
    k = int(context["target_k"])
    m = int(pool.shape[0])

    if k >= m:
        return np.arange(m, dtype=np.int64)

    anchor = samples[0]
    first = int(np.argmax(pool @ anchor))

    selected = np.empty(k, dtype=np.int64)
    selected[0] = first
    chosen_mask = np.zeros(m, dtype=bool)
    chosen_mask[first] = True

    max_sim_to_selected = pool @ pool[first]
    max_sim_to_selected[first] = np.inf

    for t in range(1, k):
        next_idx = int(np.argmin(max_sim_to_selected))
        selected[t] = next_idx
        chosen_mask[next_idx] = True

        sims = pool @ pool[next_idx]
        np.maximum(max_sim_to_selected, sims, out=max_sim_to_selected)
        max_sim_to_selected[chosen_mask] = np.inf

    return selected
