"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
from src.utils import convert_spectrogram_to_heatmap

"""Conveniance wrapper to perform STFT and iSTFT"""

from torch.fft import fftfreq
import torch.nn.functional as F
import torch as th


def cutoff(S, roll_percent=0.98, sr=8000, n_fft=512, embedding_dict=None, mask=False):

    n_freqs = S.shape[-2]
    device = S.get_device()

    # Compute the center frequencies of each bin
    # freq = fft_frequencies(sr=sr, n_fft=n_fft)
    freq = fftfreq(n_fft, 1 / sr)[:n_freqs].to(device)

    # Make sure that frequency can be broadcast
    if freq.ndim == 1:
        # reshape for broadcasting
        # freq = expand_to(freq, ndim=S.ndim, axes=-2)
        freq = freq.reshape(1,1,-1,1)

    total_energy = th.cumsum(S, dim=-2)
    # (channels,freq,frames)

    threshold = roll_percent * total_energy[..., -1, :]

    # reshape threshold for broadcasting
    threshold = th.unsqueeze(threshold, dim=-2)

    ind = th.where(total_energy < threshold, th.nan, 1)

    thresholded_freq = th.nan_to_num(ind*freq, nan=float('inf'))

    rolloff: th.Tensor = th.min(thresholded_freq, dim=-2, keepdims=True)

    if embedding_dict:
        output = embedding_dict(rolloff.indices.squeeze(dim=1)).permute(0,1,3,2)
    elif mask:
        m = th.arange(n_freqs).reshape(1, 1, -1, 1).to(device)
        m = m.expand_as(S)
        indices = rolloff.indices.to(device)
        output = th.where(m < indices, 1, 0)
    else:
        output = F.one_hot(rolloff.indices.squeeze(dim=1), num_classes=n_freqs).transpose(-1, -2).flip(-2)

    return output

def spectro(x, n_fft=512, hop_length=None, pad=0, win_length=None):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=th.hann_window(win_length or n_fft).to(x),
                win_length=win_length or n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0, win_length=None):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = win_length or n_fft // (1 + pad)
    x = th.istft(z,
                 n_fft,
                 hop_length or n_fft // 2,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)