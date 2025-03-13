from vocalseg.utils import spectrogram

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import glob2
import argparse
import librosa
import json
import os
from tqdm import tqdm
import pandas as pd
import soundfile as sf


"""
python src/plot_spectrogram_grid.py --input_folder_path=results/MarmosetVocalizations2019/parameters_v21-06-21
python src/plot_spectrogram_grid.py --input_folder_path=results_non_denoised/MarmosetVocalizations2019/parameters_v21-06-21
python src/plot_spectrogram_grid.py --input_folder_path=results/MacaqueVocalizations_2021_11_05 --filter_with_expert_labels --parameters_file=results/MacaqueVocalizations2021/2021_11_05/parameters_v03-01-22/parameters_v03-01-22_parameters.txt
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_folder_path', nargs='?', type=str, default=None)
    parser.add_argument('--parameters_file', nargs='?', type=str, default=None)
    parser.add_argument('--filter_with_expert_labels', action='store_true', default=False)
    args = parser.parse_args()

    if args.filter_with_expert_labels:
        df = pd.read_csv(os.path.join(args.input_folder_path, f'{os.path.basename(args.input_folder_path)}_expert_labels_3rd_pass.tsv'), sep='\t')
        #df['id'] = [name.replace('_denoised_n-std-thresh-2.0_prop-decrease-0.8', '') for name in df.id]
        df = df[df.type == 'Yes']
        file_paths = [os.path.join(args.input_folder_path, 'audios', f'{file_name}.wav') for file_name in df.id.tolist()]
    else:
        file_paths = glob2.glob(os.path.join(args.input_folder_path, '**/audios/*.wav'))
    spectrograms = list()
    parameters_file_path = args.parameters_file if args.parameters_file else os.path.join(args.input_folder_path, os.path.basename(args.input_folder_path) + '_parameters.txt')
    with open(parameters_file_path, 'r') as f:
        parameters = json.load(f)
    with tqdm(file_paths) as bar:
        for file_path in bar:
            samples, sample_rate = librosa.load(file_path)
            #assert(sample_rate == parameters['sr'])
            shrinkage_frame_number = int(0.4 * sample_rate)
            samples = samples[shrinkage_frame_number:-shrinkage_frame_number]
            spec = spectrogram(
                samples,
                parameters['sr'],
                n_fft=parameters['n_fft'],
                hop_length_ms=parameters['hop_length_ms'],
                win_length_ms=parameters['win_length_ms'],
                ref_level_db=parameters['ref_level_db'],
                pre=parameters['pre'],
                min_level_db=parameters['min_level_db']
            )
            spectrograms.append(spec)
            """sf.write(os.path.join(args.input_folder_path, 'audios_shrinked', os.path.basename(file_path)), samples, sample_rate)

            fig, ax = plt.subplots(figsize=(10, 15))
            sample_length = len(samples) / sample_rate
            ax.imshow(spec, interpolation='nearest', aspect="auto", origin="lower", extent=(0.0, sample_length, 0, parameters['sr'] / 2))
            yl = np.linspace(0, parameters['sr'] / 2, 5).astype(int).tolist()
            xl = np.round(np.linspace(0.0, sample_length, 5), 2).tolist()
            ax.set_xlabel('Time (s)')
            ax.set(xticks=xl, xticklabels=xl)
            ax.set_ylabel('Frequency (Hz)')
            ax.set(yticks=yl, yticklabels=yl)
            plt.savefig(os.path.join(args.input_folder_path, 'images_shrinked', os.path.splitext(os.path.basename(file_path))[0] + '.png'), bbox_inches='tight', dpi=100)
            plt.close()"""

            #bar.update(1)

    line_size = int(np.sqrt(len(spectrograms))) + 1
    fig, axes = plt.subplots(nrows=line_size, ncols=line_size, figsize=(50, 50))
    print('Subplots built.')
    with trange(len(spectrograms)) as bar:
        for i in bar:
            row = i // line_size
            col = i % line_size
            axes[row, col].imshow(spectrograms[i], interpolation='nearest', aspect="auto", origin="lower")
            bar.update(1)
    for row in range(line_size):
        for col in range(line_size):
            axes[row, col].axis("off")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.savefig('results/spectrograms_grid_MacaqueVocalizations_2021_11_05_non_denoised_100ms_3rd_pass.png', bbox_inches='tight', dpi=100)
