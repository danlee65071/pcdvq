import torch
import torch.nn.functional as F


def reshape_pq_to_k(x: torch.Tensor, k: int, pad_value=0):
    p, q = x.shape
    n = p * q
    rem = n % k

    flat = x.reshape(-1)
    if rem != 0:
        pad_elems = k - rem
        pad = flat.new_full((pad_elems,), pad_value)
        flat = torch.cat([flat, pad], dim=0)

    y = flat.view(-1, k)
    return y


def reshape_k_to_pq(y: torch.Tensor, p: int, q: int):
    n = p * q
    flat = y.reshape(-1)
    x = flat[:n].reshape(p, q)
    return x
