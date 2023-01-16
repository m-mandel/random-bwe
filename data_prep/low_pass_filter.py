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


def compute_cutoff_t(cutoff, alpha, beta, t, fs=8000):
    """

    """
    return (math.pi / cutoff + alpha * math.sin(beta * t)) * fs / 2 * math.pi


def filter_t(cutoff, alpha, beta, t, fs=8000, filter_len=8):
    half_size = int(filter_len / 2)
    cutoff = compute_cutoff_t(cutoff, alpha, beta, t, fs)
    window = torch.hann_window(2 * half_size + 1, periodic=False)
    time = torch.arange(-half_size, half_size + 1)
    filter = 2 * cutoff * window * sinc(2 * cutoff * math.pi * time)
    # Normalize filter to have sum = 1, otherwise we will have a small leakage
    # of the constant component in the input signal.
    filter /= filter.sum()

    return filter

def compute_conv_mat(n, cutoff, alpha, beta, fs=8000, filter_len=8):
    half_size = int(filter_len / 2)
    mat = filter_len((n + half_size, n + half_size))
    for t in range(mat.shape[0]):
        mat[t,t:t + filter_len] = torch.flip(filter_t(cutoff, alpha, beta, t, fs, filter_len), [0])
    return mat[:, half_size:-half_size]



class LowPassFilter(torch.nn.Module):

    def __init__(self, cutoff, fs: float = 8000, filter_len: float = 8):
        super().__init__()
        self.cutoff = cutoff
        self.fs = fs
        self.filter_len = filter_len

    def forward(self, input, alpha, beta):
        n = len(input)
        conv_mat = compute_conv_mat(n, self.cutoff, alpha, beta, self.fs, self.filter_len)
        output = torch.matmul(conv_mat, input)
        return output