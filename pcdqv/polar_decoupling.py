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
        phis = torch.zeros(num_vectors, k-1, dtype=x.dtype).to(x.device)
        magnitudes = torch.norm(x, dim=1).to(x.device)
        for i in range(k-1):
            input_atan2 = torch.sqrt(torch.sum(x[:, i+1:] ** 2, dim=1))
            other_atan2 = x[:, i]
            phi = torch.atan2(input_atan2, other_atan2)
            phis[:, i] = phi
        return phis, magnitudes
