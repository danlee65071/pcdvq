import math
import torch

from pcdqv.polar_decoupling import PCDVQ
from pcdqv.codebooks import construct_direction_codebook, construct_magnitude_codebook


def test_unit_directions_2d_matches_cos_sin():
    phis = torch.tensor([[0.0], [math.pi / 6], [math.pi / 2], [math.pi], [3 * math.pi / 2]], dtype=torch.float32)
    magnitudes = torch.tensor([1.0])
    pc = PCDVQ(directions_codebook=phis, magnitudes_codebook=magnitudes)

    expected = torch.stack([torch.cos(phis.squeeze(-1)), torch.sin(phis.squeeze(-1))], dim=-1)
    assert pc.unitary_directions.shape == expected.shape
    assert torch.allclose(pc.unitary_directions, expected, atol=1e-6)


def test_unit_directions_3d_matches_hyperspherical():
    # k = 3 -> angles (phi0, phi1), vector = [cos phi0, sin phi0 cos phi1, sin phi0 sin phi1]
    phis = torch.tensor([
        [0.3, 0.7],
        [1.0, 1.2],
        [2.3, 0.4],
    ], dtype=torch.float32)
    magnitudes = torch.tensor([1.0])
    pc = PCDVQ(directions_codebook=phis, magnitudes_codebook=magnitudes)

    phi0 = phis[:, 0]
    phi1 = phis[:, 1]
    expected = torch.stack([
        torch.cos(phi0),
        torch.sin(phi0) * torch.cos(phi1),
        torch.sin(phi0) * torch.sin(phi1),
    ], dim=-1)
    assert pc.unitary_directions.shape == expected.shape
    assert torch.allclose(pc.unitary_directions, expected, atol=1e-6)


def test_unitary_directions_are_unit_norm():
    # Random angles for k=4; resulting vectors must be unit length
    torch.manual_seed(0)
    N, k = 16, 4
    phis = torch.rand(N, k - 1, dtype=torch.float32) * math.pi
    pc = PCDVQ(directions_codebook=phis, magnitudes_codebook=torch.tensor([1.0]))
    norms = torch.norm(pc.unitary_directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_to_polar_recovers_phis_and_magnitude():
    torch.manual_seed(1)
    phis = torch.tensor([
        [0.8, 1.1],
        [1.2, 0.5],
    ], dtype=torch.float32)
    magnitudes = torch.tensor([2.5, 0.7], dtype=torch.float32)
    pc = PCDVQ(directions_codebook=phis, magnitudes_codebook=magnitudes)

    # Construct x = r * u(phis)
    u = pc.get_unit_directions(phis)
    x = u * magnitudes.unsqueeze(-1)

    phis_rec, mags_rec = PCDVQ.to_polar(x)
    assert torch.allclose(mags_rec, magnitudes, atol=1e-6)
    assert torch.allclose(phis_rec, phis, atol=1e-5)


def test_to_polar_zero_vector_edge_case():
    # Zero vector: magnitude is zero; angles default to zero by atan2(0,0)
    k = 4
    phis = torch.zeros(1, k - 1, dtype=torch.float32)
    pc = PCDVQ(directions_codebook=phis, magnitudes_codebook=torch.tensor([0.0]))
    x = torch.zeros(3, k, dtype=torch.float32)
    phis_rec, mags_rec = pc.to_polar(x)
    assert torch.allclose(mags_rec, torch.zeros(3), atol=0)
    assert torch.allclose(phis_rec, torch.zeros(3, k - 1), atol=0)


def test_forward_quantizes_exact_codebook_points():
    # Build direction codebook from random candidates, magnitude codebook from chi model
    torch.manual_seed(7)
    k = 4
    # Candidates and selected directions
    cand_phis = torch.rand(256, k - 1) * math.pi
    candidates = PCDVQ.get_unit_directions(cand_phis)
    a = 3  # 2^a directions
    selected_dirs = construct_direction_codebook(candidates, a)
    # Direction angles for PCDVQ
    phis_cb, mags_cb_unit = PCDVQ.to_polar(selected_dirs)
    assert torch.allclose(mags_cb_unit, torch.ones_like(mags_cb_unit), atol=1e-6)

    # Magnitude codebook
    mag_bits = 3
    mag_cb = construct_magnitude_codebook(mag_bits, k=k, tau=0.999, tol=1e-4, max_iters=30)

    pc = PCDVQ(directions_codebook=phis_cb, magnitudes_codebook=mag_cb)

    # Build inputs exactly at codebook combos
    idx_dirs = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    idx_mags = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    x = selected_dirs[idx_dirs] * mag_cb[idx_mags].unsqueeze(-1)

    out = pc.forward(x)
    qx = out["x_q"]
    didx = out["idx_dir"]
    midx = out["idx_rad"]
    assert torch.allclose(qx, x, atol=1e-6)
    assert torch.equal(didx, idx_dirs)
    assert torch.equal(midx, idx_mags)


def test_forward_matches_bruteforce_baseline_on_random():
    torch.manual_seed(11)
    k = 5
    # Build direction codebook
    cand_phis = torch.rand(512, k - 1) * math.pi
    candidates = PCDVQ.get_unit_directions(cand_phis)
    a = 4  # 16 directions
    selected_dirs = construct_direction_codebook(candidates, a)
    phis_cb, mags_cb_unit = PCDVQ.to_polar(selected_dirs)
    assert torch.allclose(mags_cb_unit, torch.ones_like(mags_cb_unit), atol=1e-6)

    # Magnitude codebook
    mag_bits = 4
    mag_cb = construct_magnitude_codebook(mag_bits, k=k, tau=0.999, tol=1e-4, max_iters=30)

    pc = PCDVQ(directions_codebook=phis_cb, magnitudes_codebook=mag_cb)

    # Random inputs
    N = 20
    x = torch.randn(N, k)

    # Baseline selection (mirror forward): compare normalized phis to normalized codebook phis
    mags = torch.linalg.vector_norm(x, dim=1)
    phis_x, _ = PCDVQ.to_polar(x)
    phis_norm = torch.nn.functional.normalize(phis_x, dim=-1)
    phis_cb_norm = torch.nn.functional.normalize(phis_cb, dim=-1)
    sims = phis_norm @ phis_cb_norm.T
    dir_idx_ref = sims.argmax(dim=1)
    diff = (mags.unsqueeze(-1) - mag_cb.unsqueeze(0)).abs()
    mag_idx_ref = diff.argmin(dim=1)
    qx_ref = selected_dirs[dir_idx_ref] * mag_cb[mag_idx_ref].unsqueeze(-1)

    out = pc.forward(x)
    qx = out["x_q"]
    dir_idx = out["idx_dir"]
    mag_idx = out["idx_rad"]
    assert torch.equal(dir_idx, dir_idx_ref)
    assert torch.equal(mag_idx, mag_idx_ref)
    assert torch.allclose(qx, qx_ref, atol=1e-6)


def test_to_polar_and_back_reconstructs_x_random():
    # Random vectors in R^5 should be reconstructable from (phis, r)
    torch.manual_seed(123)
    N, k = 8, 5
    x = torch.randn(N, k, dtype=torch.float32)
    phis_rec, mags_rec = PCDVQ.to_polar(x)
    u = PCDVQ.get_unit_directions(phis_rec)
    x_hat = u * mags_rec.unsqueeze(-1)
    assert torch.allclose(x_hat, x, atol=1e-5)


def test_get_unit_directions_shapes_and_dtype():
    # For input (N, k-1) returns (N, k) and preserves dtype/device
    N, k = 3, 6
    phis64 = (torch.rand(N, k - 1, dtype=torch.float64) * math.pi)
    u = PCDVQ.get_unit_directions(phis64)
    assert u.shape == (N, k)
    assert u.dtype == torch.float64


def test_to_polar_magnitude_is_euclidean_norm():
    torch.manual_seed(0)
    x = torch.randn(10, 7)
    phis, mags = PCDVQ.to_polar(x)
    assert phis.shape == (10, 6)
    expected = torch.linalg.vector_norm(x, dim=1)
    assert torch.allclose(mags, expected, atol=1e-6)


def test_to_polar_angle_ranges():
    # All but last angles in [0, pi], last in [0, 2*pi)
    torch.manual_seed(0)
    x = torch.randn(12, 6)
    phis, _ = PCDVQ.to_polar(x)
    assert torch.all(phis[:, :-1] >= 0)
    assert torch.all(phis[:, :-1] <= math.pi + 1e-6)
    assert torch.all(phis[:, -1] >= 0)
    assert torch.all(phis[:, -1] < 2 * math.pi + 1e-6)
