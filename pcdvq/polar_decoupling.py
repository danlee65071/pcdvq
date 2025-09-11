import math
import torch
from typing import Tuple, Optional

from pcdvq.utils import reshape_pq_to_k, reshape_k_to_pq
from hadamard_transform import randomized_hadamard_transform, inverse_randomized_hadamard_transform


class PCDVQ:
    def __init__(
        self,
        directions_codebook: torch.Tensor,
        magnitudes_codebook: torch.Tensor,
    ) -> None:
        self.directions_codebook = directions_codebook
        self.magnitudes_codebook = magnitudes_codebook

    @staticmethod
    def get_unit_directions(directions: torch.Tensor) -> torch.Tensor:
        sin_matrix = torch.sin(directions)
        cos_matrix = torch.cos(directions)

        # cum_sin[i] = sin(directions[i, 0]) * ... * sin(directions[i, j])
        cum_sin = torch.cumprod(sin_matrix, dim=-1)
        # sin_prefix[i] = [1, sin(directions[i, 0]), sin(directions[i, 0]) * sin(directions[i, 1]), ...,
        # sin(directions[i, 0]) * ... * sin(directions[i, k-2])]
        sin_prefix = torch.cat([torch.ones_like(sin_matrix[:, :1]), cum_sin[:, :-1]], dim=-1)

        x_except_last = sin_prefix * cos_matrix
        x_last = cum_sin[:, -1:].clone()
        unitary_directions = torch.cat([x_except_last, x_last], dim=-1)
        return unitary_directions

    @staticmethod
    def to_polar(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p, k = x.shape
        magnitudes = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        phis = torch.empty(p, k-1, dtype=x.dtype, device=x.device)
        x_squares = x * x
        x_squares_flipped = torch.flip(x_squares, dims=[1])
        x_squares_cumsum_flipped = torch.cumsum(x_squares_flipped, dim=1)
        x_squares_cumsum = torch.flip(x_squares_cumsum_flipped, dims=[1])
        y_head = x_squares_cumsum[:, 1:-1].sqrt()
        x_head = x[:, :-2]
        phis[:, :-1] = torch.atan2(y_head, x_head)
        phi_last = torch.atan2(x[:, -1], x[:, -2])
        phi_last = (phi_last + 2 * math.pi) % (2 * math.pi)
        phis[:, -1] = phi_last
        return phis, magnitudes.squeeze(-1)

    @staticmethod
    def to_cartesian(phis: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if r.ndim == 1:
            r = r.unsqueeze(-1)
        unit_dirs = PCDVQ.get_unit_directions(phis)
        x = r * unit_dirs
        return x

    def forward(self, x: torch.Tensor, seed: Optional[int] = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        
        prng = torch.Generator(device='cpu')
        prng.manual_seed(seed)
        # standard gaussian regularization
        x_sgr = randomized_hadamard_transform(x, prng=prng)
        
        # to polar
        phi_k = self.directions_codebook.shape[-1]
        phis, r = self.to_polar(x_sgr)
        phi_p, phi_q = phis.shape
        reshaped_phis = reshape_pq_to_k(phis, phi_k)
        C_phi = self.directions_codebook.to(device=device, dtype=dtype)
        z = torch.nn.functional.normalize(reshaped_phis, dim=-1)
        Z = torch.nn.functional.normalize(C_phi, dim=-1)
        sim = z @ Z.T
        idx_dir = sim.argmax(dim=1)

        C_r = self.magnitudes_codebook.to(device=device, dtype=dtype).view(-1)
        d = (r.view(-1, 1) - C_r.view(1, -1)).abs()
        idx_rad = d.argmin(dim=1)

        # reverse
        phis_q = C_phi[idx_dir]
        reshaped_phis_q = reshape_k_to_pq(phis_q, phi_p, phi_q)
        r_q = C_r[idx_rad].unsqueeze(1)
        x_q = inverse_randomized_hadamard_transform(PCDVQ.to_cartesian(reshaped_phis_q, r_q), prng=prng)

        return {
            "phis": phis,
            "r": r,
            "idx_dir": idx_dir,
            "idx_rad": idx_rad,
            "phis_q": reshaped_phis_q,
            "phi_sim": sim,
            "r_q": r_q,
            "x_q": x_q
        }
