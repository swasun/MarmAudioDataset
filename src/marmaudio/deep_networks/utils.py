# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import soundfile as sf
import torchvision.models as torchmodels
import torch
from torch import nn
from torch.utils import data
import numpy as np
from scipy.signal import resample



marmoset_labels = ['Twitter', 'Trill', 'Noise', 'Tsik', 'Infant_cry', 'Tsik-Ek',
       'Seep-Ek', 'Scream', 'Seep', 'Ek', 'Phee', 'Trill-Phee']
typetoidx_marmoset = {k:i for i, k in enumerate(marmoset_labels)}
idxtotype_marmoset = {typetoidx_marmoset[k]:k for k in typetoidx_marmoset}

voc_vs_noise_labels = ['Vocalization', 'Noise']
typetoidx_voc_vs_noise = {k:i for i, k in enumerate(voc_vs_noise_labels)}
idxtotype_voc_vs_noise = {typetoidx_voc_vs_noise[k]:k for k in typetoidx_voc_vs_noise}

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class Dataset(data.Dataset):
    def __init__(self, df, audio_path, sampleDur=.5, retType=False, brown=False, species_name='marmoset', fs=None, from_raw_recordings=True):
        super(Dataset, self)
        self.df, self.audio_path, self.retType, self.brown = df, audio_path, retType, brown
        self.target_fs = fs if not fs is None else (96000 if species_name == 'marmoset' else 44100)
        self.sampleDurSec = sampleDur
        self.sampleDur = int(sampleDur * self.target_fs)
        self.species_name = species_name
        self.from_raw_recordings = from_raw_recordings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.from_raw_recordings:
            row = self.df.iloc[idx]
            row_name = row.name
            info = sf.info(f'{self.audio_path}/{row.parent_name}.wav')
            dur, fs = info.duration, info.samplerate
            start = np.clip(row.pos - self.sampleDur//2, 0, int((dur - self.sampleDurSec)*fs) )
            sig, fs = sf.read(f'{self.audio_path}/{row.parent_name}.wav', start=start, stop=start + int(self.sampleDurSec*fs), always_2d=True)
            sig = sig[:,0]
            if len(sig) < self.sampleDurSec*fs:
                print(f'failed with {self.audio_path}/{row.parent_name}.wav - {sig.shape=} - {row.pos=} - {self.sampleDur=} - start={row.pos - self.sampleDur//2} - stop={row.pos + self.sampleDur//2}')
                return None
            if fs != self.target_fs:
                print('resampling '+str(row_name))
                sig = resample(sig, int(len(sig)/fs*self.target_fs))
        else:
            row = self.df.iloc[idx]
            row_name = row.file_id
            sig, fs = sf.read(f'{self.audio_path}/{row_name}.wav', always_2d=True)
            sig = sig[:,0]

            if fs != self.target_fs:
                print('resampling '+str(row_name))
                sig = resample(sig, int(len(sig)/fs*self.target_fs))
            
            middle_point = len(sig) // 2
            ms_samples = int(0.5 * fs)

            if 2 * middle_point < ms_samples:
                # The signal is less than 500 ms, so pad it to 500 ms with zeros
                padding = ms_samples - len(sig)
                sampled_sig = np.pad(sig, (padding // 2, padding - padding // 2), mode='constant')
            elif 2 * middle_point == ms_samples:
                # The signal is exactly 500 ms, so return the whole signal
                sampled_sig = sig
            else:
                # The signal is longer than 500 ms, so take a 500 ms sample from the middle
                start_index = middle_point - ms_samples // 2
                end_index = middle_point + ms_samples // 2
                sampled_sig = sig[start_index:end_index]

            sig = sampled_sig

        if self.brown:
            brown_std = 10**(-np.random.normal(1,1)/20) # -1 < SNR < 3
            sig = norm(sig) + norm(np.cumsum(np.random.normal(0, 1, len(sig))))*brown_std
        if self.retType:
            if self.species_name == 'marmoset':
                return torch.Tensor(norm(sig)).float(), row_name, typetoidx_marmoset[row.type]
            elif self.species_name == 'voc_vs_noise':
                return torch.Tensor(norm(sig)).float(), row_name, typetoidx_voc_vs_noise[row.type]
            else:
                raise ValueError(f'Unsupported species {self.species_name}')
        else:
            x = torch.Tensor(norm(sig)).float()
            return x, row_name


class Continuous_Dataset(data.Dataset):
    def __init__(self, audio_path, sample_dur, species_name='marmoset', fs=None):
        super(Continuous_Dataset, self)
        self.sample_dur, self.audio_path, = sample_dur, audio_path
        self.target_fs = fs if not fs is None else 96000 if species_name == 'marmoset' else 44100
        self.info = sf.info(self.audio_path)

    def __len__(self):
        return int(self.info.duration / self.sample_dur)

    def __getitem__(self, idx):
        sig, fs = sf.read(self.audio_path, start=idx * self.sample_dur * self.info.samplerate, stop=(idx + 1) * self.sample_dur * self.info.samplerate, always_2d=True)
        sig = sig[:,0]
        if fs != self.target_fs:
            sig = resample(sig, int(len(sig)/fs*self.target_fs))
        return torch.Tensor(norm(sig)).float(), idx * self.sample_dur


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

def norm(arr):
    return (arr - np.mean(arr) ) / np.std(arr)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape1dto2d(nn.Module):
    def __init__(self, outshape):
        super(Reshape1dto2d, self).__init__()
    def forward(self, x):
        return x.view()

class Reshape2dto1d(nn.Module):
    def __init__(self):
        super(Reshape2dto1d, self).__init__()
    def forward(self, x):
        return x[:,:,0,:]

class Reshape(nn.Module):
    """
    Reshape the input volume
    Args:
        *shape (ints): new shape, WITHOUT specifying batch size as first
        dimension, as it will remain unchanged.
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


def VGG():
    vgg16 = torchmodels.vgg16(weights=torchmodels.VGG16_Weights.DEFAULT)
    vgg16 = vgg16.features[:13]
    for nm, mod in vgg16.named_modules():
        if isinstance(mod, nn.MaxPool2d):
            setattr(vgg16, nm,  nn.AvgPool2d(2 ,2))
    return vgg16

class Croper2D(nn.Module):
    def __init__(self, *shape):
        super(Croper2D, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x[:,:,:self.shape[0],:self.shape[1]]
