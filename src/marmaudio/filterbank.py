# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

from torch import nn
import torch
import numpy as np


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq,
                          norm=True, crop=False):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples. If `norm`, will normalize
    each filter by its area. If `crop`, will exclude rows that exceed the
    maximum frequency and are therefore zero.
    """
    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, num_bands + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate

    # create filterbank
    input_bins = (frame_len // 2) + 1
    if crop:
        input_bins = min(input_bins,
                         int(np.ceil(max_freq * frame_len /
                                     float(sample_rate))))
    x = torch.arange(input_bins, dtype=peaks_bin.dtype)[:, np.newaxis]
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    # triangles are the minimum of two linear functions f(x) = a*x + b
    # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
    tri_left = (x - l) / (c - l)
    # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
    tri_right = (x - r) / (c - r)
    # combine by taking the minimum of the left and right sides
    tri = torch.min(tri_left, tri_right)
    # and clip to only keep positive values
    filterbank = torch.clamp(tri, min=0)

    # normalize by area
    if norm:
        filterbank /= filterbank.sum(0)

    return filterbank

class MelFilter(nn.Module):
    def __init__(self, sample_rate, winsize, num_bands, min_freq, max_freq):
        super(MelFilter, self).__init__()
        melbank = create_mel_filterbank(sample_rate, winsize, num_bands,
                                        min_freq, max_freq, crop=True)
        self.register_buffer('bank', melbank)

    def forward(self, x):
        x = x.transpose(-1, -2)  # put fft bands last
        x = x[..., :self.bank.shape[0]]  # remove unneeded fft bands
        x = x.matmul(self.bank)  # turn fft bands into mel bands
        x = x.transpose(-1, -2)  # put time last
        return x

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(MelFilter, self).state_dict(destination, prefix, keep_vars)
        # remove all buffers; we use them as cached constants
        for k in self._buffers:
            del result[prefix + k]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # ignore stored buffers for backwards compatibility
        for k in self._buffers:
            state_dict.pop(prefix + k, None)
        # temporarily hide the buffers; we do not want to restore them
        buffers = self._buffers
        self._buffers = {}
        result = super(MelFilter, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result

class STFT(nn.Module):

    def __init__(self, winsize, hopsize, complex=False):
        super(STFT, self).__init__()
        self.winsize = winsize
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(winsize, periodic=False))
        self.complex = complex

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(STFT, self).state_dict(destination, prefix, keep_vars)
        # remove all buffers; we use them as cached constants
        for k in self._buffers:
            del result[prefix + k]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # ignore stored buffers for backwards compatibility
        for k in self._buffers:
            state_dict.pop(prefix + k, None)
        # temporarily hide the buffers; we do not want to restore them
        buffers = self._buffers
        self._buffers = {}
        result = super(STFT, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result

    def forward(self, x):
        x = x.unsqueeze(1)
        # we want each channel to be treated separately, so we mash
        # up the channels and batch size and split them up afterwards
        batchsize, channels = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # we apply the STFT
        x = torch.stft(x, self.winsize, self.hopsize, window=self.window,
                       center=False, return_complex=False)
        # we compute magnitudes, if requested
        if not self.complex:
            x = x.norm(p=2, dim=-1)
        # restore original batchsize and channels in case we mashed them
        x = x.reshape((batchsize, channels, -1) + x.shape[2:]) #if channels > 1 else x.reshape((batchsize, -1) + x.shape[2:])
        return x

class Log1p(nn.Module):
    """
    Applies log(1 + 10**a * x), with scale fixed or trainable.
    """
    def __init__(self, a=0, trainable=False):
        super(Log1p, self).__init__()
        if trainable:
            a = nn.Parameter(torch.tensor(a, dtype=torch.get_default_dtype()))
        self.a = a
        self.trainable = trainable

    def forward(self, x):
        if self.trainable or self.a != 0:
            x = torch.log1p(10 ** self.a * x)
        return x

    def extra_repr(self):
        return 'trainable={}'.format(repr(self.trainable))
