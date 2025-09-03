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
