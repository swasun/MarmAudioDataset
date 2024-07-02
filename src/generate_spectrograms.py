from vocalseg.utils import butter_bandpass_filter, spectrogram

import os
import json
from scipy.io import wavfile
import glob2
from tqdm import tqdm
import pathlib
import numpy as np
from librosa import filters
import matplotlib.pyplot as plt


def plot_spectrogram(spec, sample_size, parameters, file_path):
    fig, ax = plt.subplots(figsize=(10, 15))
    sample_length = sample_size / parameters['sr']
    ax.imshow(spec, interpolation='nearest', aspect="auto", origin="lower", extent=(0.0, sample_length, 0, parameters['sr'] / 2))
    yl = np.linspace(0, parameters['sr'] / 2, 5).astype(int).tolist()
    xl = np.round(np.linspace(0.0, sample_length, 5), 2).tolist()
    ax.set_xlabel('Time (s)')
    ax.set(xticks=xl, xticklabels=xl)
    ax.set_ylabel('Frequency (Hz)')
    ax.set(yticks=yl, yticklabels=yl)
    plt.savefig(file_path, bbox_inches='tight', dpi=100)
    plt.close()

def preprocess_data(x, sample_size):
    x = np.asarray(x, dtype="float")

    # If it's too long, truncate it.
    if x.shape[0] > sample_size:
        x = x[: sample_size]

    # If it's too short, zero-pad it.
    start = (sample_size - x.shape[0]) // 2

    # Zero fill
    x_padded = np.zeros(sample_size, dtype=np.float32)
    x_padded[start : start + x.shape[0]] = x
    
    return x_padded

destination_root_path = 'results/MarmosetVocalizations_all'
destination_audio_path = os.path.join(destination_root_path, 'audios')
destination_spectrogram_path = os.path.join(destination_root_path, 'spectrograms')
destination_mel_spectrogram_path = os.path.join(destination_root_path, 'mel_spectrograms')

pathlib.Path(destination_spectrogram_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(destination_mel_spectrogram_path).mkdir(parents=True, exist_ok=True)

with open(os.path.join(destination_root_path, 'marmoset_experiment_v21-06-21_parameters.json')) as f:
    parameters = json.load(f)

file_paths = glob2.glob(os.path.join(destination_audio_path, '*.wav'))
sample_size = 48000

with tqdm(file_paths) as bar:
    for file_path in bar:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        bar.set_description(f'{file_path} - loading')
        _, data = wavfile.read(file_path)
        bar.set_description(f'{file_path} - filtering')
        data = data.astype(np.float32)
        data = butter_bandpass_filter(data, parameters['lowpass_filter'], (parameters['sr']//2)-1, parameters['sr'], order=2)
        data = preprocess_data(data, sample_size)

        spec = spectrogram(
            data,
            parameters['sr'],
            n_fft=parameters['n_fft'],
            hop_length_ms=parameters['hop_length_ms'],
            win_length_ms=parameters['win_length_ms'],
            ref_level_db=parameters['ref_level_db'],
            pre=parameters['pre'],
            min_level_db=parameters['min_level_db'],
        ).astype(np.float32)

        mel_basis = filters.mel(parameters['sr'], parameters['n_fft'], n_mels=64).astype(np.float32)
        mel_spec = np.dot(mel_basis, spec).astype(np.float32)

        plot_spectrogram(spec, sample_size, parameters, f'{basename}_spec.png')
        plot_spectrogram(mel_spec, sample_size, parameters, f'{basename}_mel_spec.png')

        np.save(os.path.join(destination_spectrogram_path, f'{basename}.npy'), spec)
        np.save(os.path.join(destination_mel_spectrogram_path, f'{basename}.npy'), mel_spec)

        bar.update(1)
