import torch
import torch.nn as nn


class QuantLinear(nn.Module):
    '''
        Quantized Linear Layer. Applies RHM, PCD, and replaces weight 
        vectors with vectors from DACC.
        https://arxiv.org/pdf/2506.05432
    '''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        a_bits: int,
        b_bits: int,
        codebook: torch.Tensor,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super(QuantLinear, self).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.a_bits: int = a_bits
        self.b_bits: int = b_bits
        self.codebook: torch.Tensor = codebook
        self.bias: bool = bias
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

    def forward(self, x):
        return x