import torch
import pytest
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from pcdvq.standard_regularization import RandomizedHadamard

def test_randomized_hadamard_inverse():
    torch.manual_seed(0)
    # Test for various sizes, including non-power-of-two
    for p in [3, 4, 7, 8, 16]:
        n_cols = 5
        x = torch.randn(p, n_cols, dtype=torch.float32)
        rh = RandomizedHadamard(p=p, seed=123, dtype=torch.float32)
        z = rh.forward(x)
        x_rec = rh.reverse(z)
        # Should recover original (within tolerance)
        assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-4), f"Failed for p={p}"

    # Test error if reverse called before forward
    rh = RandomizedHadamard(p=4)
    with pytest.raises(ValueError):
        rh.reverse(torch.randn(4, 2))

def test_randomized_hadamard_func():
    torch.manual_seed(0)
    for p in [2, 4, 8, 16]:
        n_cols = p
        x = torch.randn(p, n_cols, dtype=torch.float32)
        rh = RandomizedHadamard(p=p, seed=123, dtype=torch.float32)
        # Should recover original (within tolerance)
        assert torch.allclose(
            x,
            RandomizedHadamard.fwht(RandomizedHadamard.fwht(x)),
            atol=1e-5,
            rtol=1e-4
        ), f"Failed for p={p}"
