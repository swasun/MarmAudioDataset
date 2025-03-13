# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from tqdm import tqdm
import glob
import os
import soundfile as sf
from scipy.signal import butter, lfilter
import warnings
import numpy as np
import math
import noisereduce as nr
import pathlib
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vocalseg.utils import spectrogram


def gen_spectrogram_cmap():
    from colorspacious import cspace_convert
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    from matplotlib.colors import ListedColormap
    from scipy.ndimage import filters
    from scipy.signal import savgol_filter


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

    #plt.figure(figsize=[5, 5])
    #plt.imshow(gamut_image)


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
    #spec = frontend(torch.tensor(audio_data).to(args.device).float().unsqueeze(0)).cpu().detach().numpy().squeeze()
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

if __name__ == "__main__":
    segmented_predictions_id = 'marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13'
    samples_per_type = 100
    audio_source_base_path = f"{segmented_predictions_id}_validation_data_23-05-20_n{samples_per_type}/audios"
    audio_dest_base_path = audio_source_base_path + '_cleaned'
    image_dest_base_path = f"{segmented_predictions_id}_validation_data_23-05-20_n{samples_per_type}/images_cleaned_new_cmap"
    sampling_rate = 96000
    ramps_padding_window_duration = 60

    pathlib.Path(audio_dest_base_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dest_base_path).mkdir(parents=True, exist_ok=True)

    cmap = gen_spectrogram_cmap()

    """marmoset_room_noise_file_path = '../../../../Marmoset/marmoset_room_noise.wav'
    noise, _ = sf.read(marmoset_room_noise_file_path)"""

    for file_path in tqdm(glob.glob(os.path.join(audio_source_base_path, '*.wav'))):
        info = sf.info(file_path)
        sig, fs = sf.read(file_path)
        sig = butter_bandpass_filter(sig, 1000, (fs//2)-1, fs, order=2)
        """denoised_audio = nr.reduce_noise(audio_clip=sig, noise_clip=noise,
            prop_decrease=0.6, n_std_thresh=2.0)"""
        sig = nr.reduce_noise(y=sig, sr=fs)
        sig = apply_raised_cosine_window(sig, sampling_rate, ramps_padding_window_duration=ramps_padding_window_duration)
        sig /= 1.01 / np.abs(sig).max()
        sig = rms_normalize_audio(sig)
        sf.write(os.path.join(audio_dest_base_path, os.path.basename(file_path)), sig, fs)

        sample_length = info.duration
        yl = np.linspace(0, fs / 2, 5).astype(int).tolist()
        xl = np.round(np.linspace(0.0, sample_length, 5), 2).tolist()

        fig, ax = plt.subplots()
        ax.imshow(compute_spectrogram(sig), interpolation='nearest', aspect="auto", origin="lower", extent=(0.0, sample_length, 0, fs / 2), cmap=cmap)
        ax.set_xlabel('Time (s)')
        ax.set(xticks=xl, xticklabels=xl)
        ax.set_ylabel('Frequency (Hz)')
        ax.set(yticks=yl, yticklabels=yl)
        plt.savefig(f'{image_dest_base_path}/{os.path.basename(file_path).replace(".wav", ".png")}', bbox_inches='tight', dpi=100)
        plt.close()
