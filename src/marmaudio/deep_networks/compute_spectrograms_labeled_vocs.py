import argparse
import torch
import models
import pandas as pd
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import thread_map
import noisereduce as nr

from prepare_validation_data_stage2 import butter_bandpass_filter

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

def save_borderless(filepath, dpi=100, fig=None):
    '''Save the current image with no whitespace
    Example filepath: "myfig.png" or r"C:\myfig.pdf" 
    '''
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0,0,1,1,0,0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches = 0, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

cmap = gen_spectrogram_cmap()

def process_row(args):
    index, row = args
    file_id = row['file_id']
    prediction_type = 'Infant_cry' if row['prediction_type'] == 'Infant cry' else row['prediction_type']
    spectrogram_file_path = os.path.join(spectrogram_folder, f"{prediction_type}_{index}.png")
    if os.path.isfile(spectrogram_file_path):
        return
    audio_file_path = os.path.join(audio_folder, prediction_type, f"{file_id}.wav")

    # Load the audio file
    audio_data, sample_rate = sf.read(audio_file_path)

    audio_data = butter_bandpass_filter(audio_data, 1000, (sample_rate//2)-1, sample_rate, order=2)
    audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate)

    middle_point = len(audio_data) // 2
    ms_samples = int(0.5 * sample_rate)
    start_index = middle_point - ms_samples // 2
    end_index = middle_point + ms_samples // 2
    audio_data = audio_data[start_index:end_index]

    # Compute the spectrogram
    spectrogram = compute_spectrogram(audio_data)

    # Save the spectrogram
    fig, ax = plt.subplots(figsize=(10, 10))
    """cm = plt.cm.Greys
    cm.set_under('white')"""
    ax.imshow(spectrogram, interpolation='nearest', aspect="auto", origin="lower", cmap=cmap) #, vmin=0.001
    save_borderless(spectrogram_file_path, fig=fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    #device = torch.device(args.device)
    #frontend = models.get[f'frontend_logMel_{args.sampling_rate_khz}'].to(device)

    segmented_predictions_id = 'marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13'
    tsv_file = f"{segmented_predictions_id}_all.tsv"
    df = pd.read_csv(tsv_file, sep='\t')
    df = df[df['prediction_type'] != 'Vocalization']

    audio_folder = f"{segmented_predictions_id}/audios"
    spectrogram_folder = f"{segmented_predictions_id}/spectrograms_cleaned_new_cmap"

    # Create the spectrogram folder if it doesn't exist
    if not os.path.exists(spectrogram_folder):
        os.makedirs(spectrogram_folder)

    # Define the number of threads for parallel execution
    num_threads = 10

    # Convert df.iterrows() to a list
    rows = list(df.iterrows())

    # Use thread_map to parallelize the processing of rows
    list(thread_map(process_row, rows, max_workers=num_threads))
