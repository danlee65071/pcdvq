import math
import torch

from pcdqv.polar_decoupling import PCDVQ
from pcdqv.codebooks import (
    construct_direction_codebook,
    construct_magnitude_codebook,
    find_max_r_bisection,
)


def test_construct_direction_codebook_basic_properties():
    torch.manual_seed(0)
    N, k, a = 256, 4, 4  # 2^a = 16 directions to pick
    phis = torch.rand(N, k - 1) * math.pi
    base_dirs = PCDVQ.get_unit_directions(phis)  # (N, k)

    selected = construct_direction_codebook(base_dirs, a)

    # Shape and membership
    assert selected.shape == (2 ** a, k)
    # Every selected vector equals some base vector (function selects by indexing)
    # We check via broadcasting closeness
    diffs = (selected.unsqueeze(1) - base_dirs.unsqueeze(0)).abs().max(dim=-1).values
    assert torch.all((diffs < 1e-7).any(dim=1))

    # Pairwise cosine similarities (off-diagonal) should be below 0.99 for diversity
    sims = (selected @ selected.T).clamp(-1, 1)
    sims = sims - torch.eye(selected.size(0))  # zero out diagonal
    max_offdiag = sims.abs().max().item()
    assert max_offdiag < 0.9999


def test_construct_magnitude_codebook_monotonic_and_within_bounds():
    bits, k, tau = 4, 6, 0.999
    tol, max_iters = 1e-4, 50

    mags = construct_magnitude_codebook(bits, k, tau, tol, max_iters)

    assert mags.shape == (2 ** bits,)
    # Strictly increasing and positive
    diffs = mags[1:] - mags[:-1]
    assert torch.all(diffs > 0)
    assert mags.min() >= 0

    # Upper bound
    max_r = find_max_r_bisection(k, tau)
    assert mags.max().item() <= max_r + 1e-6

    # Deterministic for fixed args
    mags2 = construct_magnitude_codebook(bits, k, tau, tol, max_iters)
    assert torch.allclose(mags, mags2, atol=1e-7)


def test_pipeline_with_constructed_codebooks_matches_unit_vectors():
    # Build candidate directions, pick a codebook, map to angles, then rebuild via PCDVQ
    torch.manual_seed(42)
    N, k, a = 512, 5, 5  # pick 32 directions
    phis_candidates = torch.rand(N, k - 1) * math.pi
    candidates = PCDVQ.get_unit_directions(phis_candidates)

    selected_dirs = construct_direction_codebook(candidates, a)
    # Convert selected unit vectors to angular representation using to_polar
    phis_cb, mags_cb = PCDVQ.to_polar(selected_dirs)
    # Magnitudes should be ~1 for unit vectors
    assert torch.allclose(mags_cb, torch.ones_like(mags_cb), atol=1e-6)

    # Build magnitude codebook (not directly used here, but ensure callable)
    _ = construct_magnitude_codebook(bits_for_magnitude=3, k=k, tau=0.999, tol=1e-4, max_iters=30)

    # Instantiate PCDVQ with the direction angles from selected directions
    pc = PCDVQ(directions_codebook=phis_cb, magnitudes_codebook=torch.tensor([1.0]))
    # unitary_directions should reproduce the selected unit vectors
    assert torch.allclose(pc.unitary_directions, selected_dirs, atol=1e-6)
