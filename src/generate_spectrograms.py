# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import argparse
import os
import soundfile as sf
from vocalseg.utils import spectrogram
from scipy.signal import butter, lfilter
import glob2
from tqdm import tqdm
import pathlib
import numpy as np
from librosa import filters
import matplotlib.pyplot as plt
import pandas as pd
import noisereduce as nr
import warnings
from colorspacious import cspace_convert
import math
from matplotlib.colors import ListedColormap
from scipy.ndimage import filters
from scipy.signal import savgol_filter
from multiprocessing import Pool


def gen_spectrogram_cmap():
    '''
    J = lightness
    C = chroma
    h = hue
    '''

    # Resolution of colorspace
    J_RES = 256
    C_RES = 256

    # NAME = 'So normal'
    # ANGLE = np.pi * 2 * 0.7
    # OFFSET = np.pi * 2 * 0.64
    # CCW = False
    # SMOOTH = 1/3

    # NAME = 'Wow unique'
    # ANGLE = np.pi * 2 * 1.0
    # OFFSET = np.pi * 2 * 0.275
    # CCW = True
    # SMOOTH = 1/2

    # NAME = 'Viridis-like (red bg)'
    # ANGLE = np.pi * 2 * 1.0
    # OFFSET = np.pi * 2 * 0.1
    # CCW = True
    # SMOOTH = 1/4

    # NAME = 'Viridis-like (purple bg)'
    # ANGLE = np.pi * 2 * 0.9
    # OFFSET = np.pi * 2 * 0.1
    # CCW = True
    # SMOOTH = 1/5

    NAME = 'Audacity proposal'
    ANGLE = np.pi * 2 * 0.875
    OFFSET = np.pi * 2 * 0.5
    CCW = False
    SMOOTH = 1/3

    DESATURATE = 0.9


    # Generate CAM02-UCS(Jp, ap, bp) colorspace
    j_space = np.linspace(0.1, 99, J_RES)
    c_space = np.linspace(0, 50, C_RES)

    if CCW:
        h_ = np.linspace(ANGLE+OFFSET, OFFSET, J_RES)
    else:
        h_ = np.linspace(OFFSET, ANGLE+OFFSET, J_RES)

    jpapbp = np.zeros([C_RES, J_RES, 3])
    for jdx, jp in enumerate(j_space):
        for cdx, chroma in enumerate(c_space):
            ap = np.cos(h_[jdx]) * chroma
            bp = np.sin(h_[jdx]) * chroma
            jpapbp[cdx, jdx] = (jp, ap, bp)

    # Convert to sRGB
    rgb = cspace_convert(jpapbp, "CAM02-UCS", "sRGB1")


    # Get chroma limit of sRGB
    c_limit = np.zeros_like(j_space)
    for jdx in range(J_RES):
        max_cdx = 0
        for cdx in range(1, C_RES):
            if np.any(rgb[cdx, jdx] <= 0) or np.any(1 < rgb[cdx, jdx]):
                max_cdx = cdx - 1
                break
            
        c_limit[jdx] = max_cdx


    # Smooth chroma limit contour
    c_smoothed = np.concatenate([-c_limit[::-1][:-1], c_limit, -c_limit[::-1][1:]])

    c_smoothed = savgol_filter(c_smoothed, math.ceil(J_RES*SMOOTH*1.5/2)*2 - 1, 3)
    c_smoothed = filters.uniform_filter1d(c_smoothed, int(J_RES*SMOOTH*1.5/2)) * DESATURATE

    c_smoothed = c_smoothed[J_RES:2*J_RES]

    c_selected = c_smoothed.clip(min=0).astype(int)


    # Generate and plot gaumt
    gamut_image = np.copy(rgb)
    gamut_image[gamut_image<=0] = 1
    gamut_image[1<gamut_image] = 0

    # Mark smoothed contour on image
    for jdx, max_c in enumerate(c_selected):
        if 0 == jdx % 2:
            gamut_image[max_c, jdx] = 1
        else:
            gamut_image[max_c, jdx] = 0

    # Get colors on contour
    cm_jpapbp = []
    for jdx, cdx in enumerate(c_smoothed):
        chroma = cdx * 50 / C_RES
        jp = j_space[jdx]
        ap = np.cos(h_[jdx]) * chroma
        bp = np.sin(h_[jdx]) * chroma

        cm_jpapbp.append([jp, ap, bp])

    cm_rgb = cspace_convert(cm_jpapbp, "CAM02-UCS", "sRGB1")
    cm_data = np.clip(cm_rgb, 0, 1)

    return ListedColormap(cm_data, name=NAME)

def compute_spectrogram(audio_data):
    spec = spectrogram(
        audio_data,
        96000,
        n_fft=1024,
        hop_length_ms=1,
        win_length_ms=4,
        ref_level_db=20,
        pre=0.97,
        min_level_db=-70,
    )
    return spec

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    if highcut > int(fs / 2):
        warnings.warn("Highcut is too high for bandpass filter. Setting to nyquist")
        highcut = int(fs / 2)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def raised_cosine(x, mu, s):
    return 1 / 2 / s * (1 + np.cos((x - mu) / s * math.pi)) * s

def apply_raised_cosine_window(audio, sr, ramps_padding_window_duration):
    window_len = int(sr * 0.001 * ramps_padding_window_duration)
    left_raised_cosine_window = raised_cosine(np.arange(window_len), window_len, window_len) # Compute a small positive raised cosine function for left silence
    right_raised_cosine_window = raised_cosine(np.arange(window_len), 0, window_len) # Compute a small negative raised cosine function for right silence
    new_audio = audio.copy()
    new_audio[0:window_len] *= left_raised_cosine_window
    new_audio[len(new_audio)-window_len:len(new_audio)] *= right_raised_cosine_window
    return new_audio

def rms_normalize_audio(audio, rms_value=0.01):
    """
    RMS normalize an audio segment so that sqrt(mean(x_i**2))==rms_value
    (per frame normalization on the energy, depending of the sample)
    Inputs
    ------
    audio : numpy array [d]
    flattened audio signal, mono
    rms_value : float
    desired rms value
    Returns
    -------
    norm_audio : numpy array [d]
    rms normalized audio
    """
    assert len(audio.shape)==1, 'Only implmented for mono audio'

    rms_audio = np.sqrt(np.mean(audio**2))
    norm_audio = (rms_value/rms_audio)*audio
    return norm_audio

def save_borderless(filepath, dpi=100, fig=None):
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0,0,1,1,0,0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches = 0, bbox_inches='tight', dpi=dpi, transparent=True)
    plt.close(fig)

def load_audio_file_and_compute_spectrogram(file_path):
    sig, fs = sf.read(file_path)
    sig = butter_bandpass_filter(sig, 200, (fs//2)-1, fs, order=2)
    sig = nr.reduce_noise(y=sig, sr=fs)
    sig = apply_raised_cosine_window(sig, sampling_rate, ramps_padding_window_duration=ramps_padding_window_duration)
    sig /= 1.01 / np.abs(sig).max()
    sig = rms_normalize_audio(sig)

    return compute_spectrogram(sig)

def compute_spectrogram_job(file_path):
    basename = os.path.splitext(os.path.basename(file_path))[0]
    output_file_name = os.path.join(destination_spectrogram_path, f'{basename}.png')
    if os.path.isfile(output_file_name):
        return True

    spectrogram = load_audio_file_and_compute_spectrogram(file_path)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(spectrogram, interpolation='nearest', aspect="auto", origin="lower", cmap=cmap)
    save_borderless(output_file_name, fig=fig)
    plt.close(fig)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--encodings_file_path', nargs='?', type=str, required=True)
    parser.add_argument('--dataset_root_path', nargs='?', type=str, required=True)
    args = parser.parse_args()

    encodings = np.load(args.encodings_file_path, allow_pickle=True).item()
    idxs, umap = encodings['idx'], encodings['umap']
    tsv_file = f"{args.dataset_root_path}/Annotations.tsv"
    df = pd.read_csv(tsv_file, sep='\t')
    idxs = [os.path.basename(idx) for idx in idxs]
    df = df[df['file_name'].isin(idxs)].reset_index(drop=True)

    source_audio_path = os.path.join(args.dataset_root_path, 'Vocalizations')
    destination_spectrogram_path = os.path.join(args.dataset_root_path, 'spectrograms')
    pathlib.Path(destination_spectrogram_path).mkdir(parents=True, exist_ok=True)

    file_paths = glob2.glob(os.path.join(source_audio_path, '*', '*.flac'))
    cmap = gen_spectrogram_cmap()
    sampling_rate = 96000
    ramps_padding_window_duration = 60

    with Pool(processes=30) as p, tqdm(total=len(file_paths)) as pbar:
        for result in p.imap(compute_spectrogram_job, file_paths):
            pbar.update()
            pbar.refresh()
