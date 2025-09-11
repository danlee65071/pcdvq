from itertools import combinations, product
import math
import numpy as np
import scipy.special
from scipy import stats
import torch
import torch.nn.functional as F


def e8_minimal_directions(device=None, dtype=torch.float16, normalize=False):
    eye_matrix = torch.eye(8, dtype=dtype, device=device)
    row_idxs, col_idxs = torch.triu_indices(8, 8, offset=1, dtype=torch.int8, device=device)
    e1, e2 = eye_matrix[row_idxs], eye_matrix[col_idxs]
    int_roots = torch.cat([e1 + e2, e1 - e2, -e1 + e2, -e1 - e2], 0)

    signs = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
    halfs = torch.cartesian_prod(*([signs] * 8)).reshape(-1, 8)
    half_roots = 0.5 * [halfs.prod(dim=-1) > 0]
    
    e8_roots = torch.cat([int_roots, half_roots], dim=0)    
    if normalize:
        e8_roots = e8_roots / math.sqrt(2.0)
    return e8_roots


def construct_direction_codebook(a, seed=42, dtype=torch.float32, device='cpu'):
    num_centers = 2 ** a
    directions = e8_minimal_directions(dtype, device)
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    rand_candidate_idx = torch.randint(0, directions.size(0), (1,), generator=g, device=device).item()
    candidates_idx = [rand_candidate_idx]
    for _ in range(1, num_centers):
        codebook = directions[candidates_idx]
        sims = directions @ codebook.T
        max_sim, _ = sims.max(dim=1)
        max_sim[candidates_idx] = torch.finfo(dtype).max
        next_candidate_idx = torch.argmin(max_sim).item()
        candidates_idx.append(next_candidate_idx)
    return directions[candidates_idx]


def find_max_r_bisection(k, tau):
    upper_bound = 2 * np.sqrt(k)
    lower_bound = 0.0
    for _ in range(20):
        mid = (upper_bound + lower_bound) / 2
        p = stats.chi2.cdf(mid ** 2, df=k)
        if p < tau:
            lower_bound = mid
        else:
            upper_bound = mid
    return upper_bound


def construct_magnitude_codebook(bits_for_magnitude, k, tau, tol, max_iters):
    num_centers = 2 ** bits_for_magnitude
    max_r = find_max_r_bisection(k, tau)
    codebook = torch.linspace(0, max_r, num_centers + 1)
    codebook = 0.5 * (codebook[:-1] + codebook[1:])
    
    for _ in range(max_iters):
        u = torch.empty(num_centers)
        u[0] = 0.0
        u[-1] = max_r
        u[1: -1] = 0.5 * (codebook[1:-1] + codebook[2:])
        max_loss = 0.0
        codebook_tmp = codebook.clone()
        for i in range(1, num_centers):
            num = scipy.special.gammainc((k + 1) / 2, u[i] ** 2 / 2) - \
                scipy.special.gammainc((k + 1) / 2, u[i - 1] ** 2 / 2)
            den = scipy.special.gammainc(k / 2, u[i] ** 2 / 2) - \
                scipy.special.gammainc(k / 2, u[i - 1] ** 2 / 2)
            cur = np.sqrt(2) * scipy.special.gamma((k + 1) / 2) / scipy.special.gamma(k / 2) * (num / (den + 1e-12))
            max_loss = max(max_loss, np.abs(cur - codebook[i]))
            codebook_tmp[i] = cur
        codebook = codebook_tmp
        if max_loss < tol:
            break
    return codebook