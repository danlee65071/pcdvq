import math
import torch

from typing import Tuple


class PCDVQ:
    def __init__(
        self,
        directions_codebook: torch.Tensor,
        magnitudes_codebook: torch.Tensor,
    ) -> None:
        self.directions_codebook = directions_codebook
        self.magnitudes_codebook = magnitudes_codebook
        
        self.unitary_directions = self._unit_directions(directions_codebook)

    def _unit_directions(self, directions: torch.Tensor) -> torch.Tensor:
        sin_matrix = torch.sin(directions)
        cos_matrix = torch.cos(directions)

        # cum_sin[i, j] = sin(directions[i, 0]) * ... * sin(directions[i, j])
        cum_sin = torch.cumprod(sin_matrix, dim=-1)
        # sin_prefix[i] = [1, sin(directions[i, 0]), sin(directions[i, 0]) * sin(directions[i, 1]), ...,
        # sin(directions[i, 0]) * ... * sin(directions[i, k-2])]
        sin_prefix = torch.cat([torch.ones_like(sin_matrix[:, :1]), cum_sin[:, :-1]], dim=-1)

        x_except_last = sin_prefix * cos_matrix
        x_last = cum_sin[:, -1:].clone()
        unitary_directions = torch.cat([x_except_last, x_last], dim=-1)
        return unitary_directions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_vectors, k = x.shape
        magnitudes = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        phis = torch.empty(num_vectors, k-1, dtype=x.dtype, device=x.device)
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
