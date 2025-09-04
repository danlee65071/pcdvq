import math
import torch

from pcdqv.polar_decoupling import PCDVQ


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


def test_forward_recovers_phis_and_magnitude():
    torch.manual_seed(1)
    phis = torch.tensor([
        [0.8, 1.1],
        [1.2, 0.5],
    ], dtype=torch.float32)
    magnitudes = torch.tensor([2.5, 0.7], dtype=torch.float32)
    pc = PCDVQ(directions_codebook=phis, magnitudes_codebook=magnitudes)

    # Construct x = r * u(phis)
    u = pc._unit_directions(phis)
    x = u * magnitudes.unsqueeze(-1)

    phis_rec, mags_rec = pc.forward(x)
    assert torch.allclose(mags_rec, magnitudes, atol=1e-6)
    assert torch.allclose(phis_rec, phis, atol=1e-5)


def test_forward_zero_vector_edge_case():
    # Zero vector: magnitude is zero; angles default to zero by atan2(0,0)
    k = 4
    phis = torch.zeros(1, k - 1, dtype=torch.float32)
    pc = PCDVQ(directions_codebook=phis, magnitudes_codebook=torch.tensor([0.0]))
    x = torch.zeros(3, k, dtype=torch.float32)
    phis_rec, mags_rec = pc.forward(x)
    assert torch.allclose(mags_rec, torch.zeros(3), atol=0)
    assert torch.allclose(phis_rec, torch.zeros(3, k - 1), atol=0)

