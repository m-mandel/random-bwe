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
    # print(f'creating mat of shape: {(n, n + filter_size)}')
    # mat = torch.zeros((n, n + filter_size))
    # print(f'mat shape: {mat.shape}. inserting filter...')
    try:
        idxs = [[], []]
        values = []
        for t in range(n):
            filter_values = torch.flip(filter_t(cutoff_ratio, alpha, beta, t, fs, zeros), [0])
            if t < half_size:
                curr_filter_size = half_size + t
                filter_values = filter_values[-curr_filter_size:]
                curr_idxs = [[t] * (curr_filter_size), [i for i in range(curr_filter_size)]]
            elif t > n - filter_size:
                curr_filter_size = n - t
                filter_values = filter_values[:curr_filter_size]
                curr_idxs = [[t] * (curr_filter_size), [i for i in range(t, t + curr_filter_size)]]
            else:
                curr_idxs = [[t] * filter_size, [i for i in range(t, t + filter_size)]]
            idxs[0].extend(curr_idxs[0])
            idxs[1].extend(curr_idxs[1])
            values.extend(filter_values)
            # mat[t, t:t + filter_size] = torch.flip(filter_t(cutoff_ratio, alpha, beta, t, fs, zeros), [0])
        mat = torch.sparse_coo_tensor(idxs, values, (n, n))
    except Exception as err:
        print(f'An exception occured at {t}/{n-1}')
        print(err)
        raise
    # return mat[:, half_size:-half_size-1]
    return mat



class LowPassFilter(torch.nn.Module):

    def __init__(self, cutoff_ratio, fs: float = 8000, zeros: float = 8):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.fs = fs
        self.zeros = zeros
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input, alpha, beta):
        n = input.shape[-1]
        if alpha == 0 or beta == 0:
            filter = filter_t(self.cutoff_ratio, alpha, beta, 0, self.fs, self.zeros) # this is a constant cutoff.
            output = F.conv1d(input.view(1,1,-1), filter.view(1,1,-1), padding='same').squeeze(dim=0)
        else:
            conv_mat = compute_conv_mat(n, self.cutoff_ratio, alpha, beta, self.fs, self.zeros)
            # conv_mat = conv_mat.to(self.device)
            # input = input.to(self.device)
            # output = torch.matmul(conv_mat, input.T).T
            output = torch.sparse.mm(conv_mat, input.T).T
            del conv_mat
        return output
        # return output.to('cpu')
