# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from marmaudio.denoise import denoise

import pandas as pd
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as T
import numpy as np


def make_spectrogram(waveform):
    spectrogram = T.Spectrogram(n_fft=512)
    spec = spectrogram(torch.from_numpy(waveform))
    return spec

def plot_spectrogram(specgram, title=None, ylabel="Frequency (kHz)", xlabel="Time (s)", sampling_rate=96000, ax=None):
    if title is not None:
        ax.set_title(title)

    num_frames, num_bins = specgram.shape
    times = np.arange(num_frames) * (num_bins / float(sampling_rate))
    frequencies = np.linspace(0, sampling_rate / 2000, num=specgram.shape[0], endpoint=False)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest",
              extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
    ax.set_aspect('auto')


if __name__ == "__main__":
    df = pd.read_csv('Annotations.tsv', sep='\t') # Read the annotations

    random_row = df.sample(n=1) # Randomly sample a line for example purpose
    file_id = random_row.index[0]

    file_path = os.path.join('Vocalizations', random_row["parent_name"].item(), random_row["file_name"].item())

    signal, sampling_rate = sf.read(file_path) # Read the vocalization waveform and store it as 'signal'

    denoised_signal = denoise(signal, sampling_rate)

    spec = make_spectrogram(signal)
    denoised_spec = make_spectrogram(denoised_signal)

    fig, axes = plt.subplots(2, 1)
    plot_spectrogram(spec, title=random_row.file_name.item(), ax=axes[0])
    plot_spectrogram(denoised_spec, title='Denoised', ax=axes[1])
    fig.tight_layout()
    plt.show()
