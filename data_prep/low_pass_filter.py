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


def compute_cutoff_t(cutoff_ratio, alpha, beta, t, fs=8000):
    """

    """
    return (math.pi / cutoff_ratio + alpha * math.sin(beta*t/fs)) / (2 * math.pi)


def filter_t(cutoff_ratio, alpha, beta, t, fs=8000, zeros=8):
    half_size = int(zeros * cutoff_ratio / 2)
    cutoff_t = compute_cutoff_t(cutoff_ratio, alpha, beta, t, fs)
    window = torch.hann_window(2 * half_size + 1, periodic=False)
    time = torch.arange(-half_size, half_size + 1)
    filter = 2 * cutoff_t * window * sinc(2 * cutoff_t * math.pi * time)
    # Normalize filter to have sum = 1, otherwise we will have a small leakage
    # of the constant component in the input signal.
    filter /= filter.sum()

    return filter

def compute_conv_mat(n, cutoff_ratio, alpha, beta, fs=8000, zeros=8):
    half_size = int(zeros * cutoff_ratio / 2)
    filter_size = half_size * 2 + 1
    mat = torch.zeros((n, n + filter_size))
    for t in range(mat.shape[0]):
        mat[t, t:t + filter_size] = torch.flip(filter_t(cutoff_ratio, alpha, beta, t, fs, zeros), [0])
    return mat[:, half_size:-half_size-1]



class LowPassFilter(torch.nn.Module):

    def __init__(self, cutoff_ratio, fs: float = 8000, zeros: float = 8):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.fs = fs
        self.zeros = zeros

    def forward(self, input, alpha, beta):
        n = input.shape[-1]
        conv_mat = compute_conv_mat(n, self.cutoff_ratio, alpha, beta, self.fs, self.zeros)
        output = torch.matmul(conv_mat, input.T).T
        return output