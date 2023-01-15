"""
Based on the Julius library Low pass filter: https://adefossez.github.io/julius/julius/lowpass.html
"""
import torch
import math
from torch.nn import functional as F

def sinc(x: torch.Tensor):
   """
   Implementation of sinc, i.e. sin(x) / x
   __Warning__: the input is not multiplied by `pi`!
   """
   return torch.where(x == 0, torch.tensor(1., device=x.device, dtype=x.dtype), torch.sin(x) / x)


# this is used for filtering in the frequency domain.
def _compl_mul_conjugate(a: torch.Tensor, b: torch.Tensor):
    """
    Given a and b two tensors of dimension 4
    with the last dimension being the real and imaginary part,
    returns a multiplied by the conjugate of b, the multiplication
    being with respect to the second dimension.
    """
    # PyTorch 1.7 supports complex number, but not for all operations.
    # Once the support is widespread, this can likely go away.

    op = "bcft,dct->bdft"
    return torch.stack([
        torch.einsum(op, a[..., 0], b[..., 0]) + torch.einsum(op, a[..., 1], b[..., 1]),
        torch.einsum(op, a[..., 1], b[..., 0]) - torch.einsum(op, a[..., 0], b[..., 1])
    ],
                       dim=-1)

class LowPassFilter(torch.nn.Module):

    def __init__(self, cutoff, stride: int = 1, pad: bool = True, zeros: float = 8):
        super().__init__()
        self.cutoff = cutoff
        self.stride = stride
        self.pad = pad
        self.zeros = zeros
        self.half_size = int(zeros / min([c for c in self.cutoffs if c > 0]) / 2)
        window = torch.hann_window(2 * self.half_size + 1, periodic=False)
        time = torch.arange(-self.half_size, self.half_size + 1)

        self.filter = 2 * cutoff * window * sinc(2 * cutoff * math.pi * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        self.filter /= self.filter.sum()

    def forward(self, input):
        if self.pad:
            input = F.pad(input, (self.half_size, self.half_size), mode='replicate')
        out = F.conv1d(input, self.filter, stride=self.stride)
        return out