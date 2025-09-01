import torch
import torch.nn as nn


class QuantLinear(nn.Module):
    def __init__(self, a_bits, b_bits):
        '''
        Quantized Linear Layer. 
        '''
        super(QuantLinear, self).__init__()

        self.a_bits: int = a_bits
        self.b_bits: int = b_bits

    def forward(self, x):
        return x