import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class StandardRegularization(ABC):
    """
    Abstract base for regularization/transformation blocks applied to tensors.

    Subclasses should implement `forward(x)` which takes a tensor and returns
    a transformed tensor of the same shape (unless otherwise documented).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the regularization/transform to `x`. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of `forward`. Must be implemented by subclasses when invertible."""
        raise NotImplementedError


class RandomizedHadamard(StandardRegularization):
    """
    Randomized Hadamard transform per column for standard Gaussian regularization.
    
    TODO: Test with permutation matrix.
    """
    def __init__(
        self,
        p: int,
        seed: int = 42,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()

        self.p = p
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.eps = torch.finfo(dtype).eps
        self.s = None
        
        # generator for reproducibility
        g = None
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
        
        # padding to next power of two
        self.n = 1 << (p - 1).bit_length()
        
        # generate random sign vector
        self.signs = (torch.randint(0, 2, (self.n,), generator=g, device=device) * 2 - 1).to(dtype)
        # TODO: here we can add permute functionality

    @staticmethod
    def fwht(x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh–Hadamard Transform.

        Expects the size of the last dimension to be a power of two. No normalization
        is applied here; caller can scale by sqrt(n) as needed.
        
        Reference: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
        """
        p = x.shape[0]
        if p & (p - 1) != 0:
            raise ValueError(f"x.size()[0] must be power of two, got x.size()[0]={p}")

        y = x.clone()
        h = 1
        while h < p:
            y0 = y[0: p: 2 * h, :]
            y1 = y[h: p: 2 * h, :]
            t = y0.clone()
            y[0: p: 2 * h, :] = t + y1
            y[h: p: 2 * h, :] = t - y1
            h <<= 1
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the randomized Hadamard transform to each column of `x`.
        
        Args:
            x: Input tensor.
        
        Returns:
            Transformed tensor.
        """
        p = x.shape[0]

        # pad x to next power of two
        if p < self.n:
            x = F.pad(x, (0, 0, 0, self.n - p))
        else:
            x = x
        
        # calcualte scaling factor for standardization
        sqrt_num_cols = math.sqrt(self.n)
        self.s = (torch.linalg.vector_norm(x, dim=0).clamp_min(self.eps) / sqrt_num_cols).unsqueeze(0)

        # apply randomized Hadamard transform
        y = self.fwht(x) / sqrt_num_cols
        y_rand = y * self.signs
        
        # TODO: apply permutation here if needed
        
        # scaling
        z = y_rand / self.s
        return z
        
    def reverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the randomized Hadamard transform.
        
        Args:
            z: Transformed tensor.
        
        Returns:
            Original tensor before transformation.
        """
        if self.s is None:
            raise ValueError("Must call forward() before reverse().")
        
        # rescale
        y_rand = z * self.s
        
        # TODO: inverse permutation here if applied in forward
        
        # inverse randomized Hadamard transform
        sqrt_num_cols = math.sqrt(self.n)
        x = self.fwht(y_rand) / sqrt_num_cols
        x = x * self.signs.view(-1, 1)
        
        # remove padding if applied
        return x[:self.p, :]
