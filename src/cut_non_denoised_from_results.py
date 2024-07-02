# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

from vocalseg.utils import butter_bandpass_filter, spectrogram

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from scipy.io import wavfile
import argparse
import pandas as pd
from tqdm import tqdm
import json
import glob2


"""
python src/cut_non_denoised_from_results.py --input_results_folder_path=results/MarmosetVocalizations2019 --input_chunked_folder_path=data/MarmosetVocalizations2019/chunked --experiment_name=parameters_v21-06-21 --output_folder_path=results_non_denoised/MarmosetVocalizations2019
python src/cut_non_denoised_from_results.py --input_results_folder_path=results/MarmosetVocalizations2020 --input_chunked_folder_path=data/MarmosetVocalizations2019/chunked --experiment_name=parameters_v21-06-21 --output_folder_path=results_non_denoised/MarmosetVocalizations2019
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_results_folder_path', nargs='?', type=str, default=None)
    parser.add_argument('--parameter_file', nargs='?', type=str, required=True, help='Parameter file path to override the default parameters without using argparse.')
    parser.add_argument('--input_chunked_folder_path', nargs='?', type=str, default=None)
    parser.add_argument('--input_chunked_file_path', nargs='?', type=str, default=None)
    parser.add_argument('--output_folder_path', nargs='?', type=str, default='results', help='Root path of the output folder.')
    parser.add_argument('--exception_files', nargs='+', type=str, default=None)
    parser.add_argument('--experiment_name', nargs='?', type=str, required=True, help='Name of the experiment.')
    args = parser.parse_args()

    if not os.path.isfile(args.parameter_file):
        raise ValueError(f"Parameter file not found '{args.parameter_file}'")
    with open(args.parameter_file) as f:
        parameters = json.load(f)

    if (not args.input_chunked_folder_path and not args.input_chunked_file_path) or \
        (args.input_chunked_folder_path and args.input_chunked_file_path):
        raise ValueError('Use either input_chunked_folder_path or input_chunked_file_path')

    if args.input_chunked_folder_path:
        file_paths = glob2.glob(os.path.join(args.input_chunked_folder_path, '**/*.wav'))
    else:
        file_paths = [args.input_chunked_file_path]

    pathlib.Path(os.path.join(args.output_folder_path, parameters['experiment_name'])).mkdir(parents=True, exist_ok=True)

    with tqdm(file_paths) as bar:
        for file_path in bar:
            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            if args.exception_files and base_file_name in args.exception_files:
                bar.set_description(f'Skipping file {base_file_name}...')
                bar.update(1)
                continue
            else:
                bar.set_description(base_file_name)

            denoised_str = '_denoised_n-std-thresh-2.0_prop-decrease-0.8'
            df = pd.read_csv(os.path.join(args.input_results_folder_path, args.experiment_name, base_file_name + denoised_str, base_file_name + denoised_str + '_labels.tsv'), sep='\t')

            print('Creating the folder tree...')
            exported_results_dir = os.path.join(args.output_folder_path, parameters['experiment_name'], base_file_name)
            audios_dir = os.path.join(exported_results_dir, 'audios')
            images_dir = os.path.join(exported_results_dir, 'images')
            pathlib.Path(audios_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(images_dir).mkdir(parents=True, exist_ok=True)

            print(f"Loading wavfile {file_path}...")
            _, data = wavfile.read(file_path)

            print(f'Filtering data using a bandpass filter...')
            data = butter_bandpass_filter(data, parameters['lowpass_filter'], (parameters['sr']//2)-1, parameters['sr'], order=2)

            with tqdm(total=df.shape[0]) as bar_audio:
                for i, row in df.iterrows():
                    onset = row['onset']
                    offset = row['offset']
                    sample = data[onset:offset]
                    file_name = f'{base_file_name}_sample_{i}'
                    bar_audio.set_description(file_name)
                    wavfile.write(os.path.join(audios_dir, f'{file_name}.wav'), parameters['sr'], data=sample)

                    spec = spectrogram(
                        sample,
                        parameters['sr'],
                        n_fft=parameters['n_fft'],
                        hop_length_ms=parameters['hop_length_ms'],
                        win_length_ms=parameters['win_length_ms'],
                        ref_level_db=parameters['ref_level_db'],
                        pre=parameters['pre'],
                        min_level_db=parameters['min_level_db'],
                    )

                    fig, ax = plt.subplots(figsize=(10, 15))
                    sample_length = (offset - onset) / parameters['sr']
                    ax.imshow(spec, interpolation='nearest', aspect="auto", origin="lower", extent=(0.0, sample_length, 0, parameters['sr'] / 2))
                    yl = np.linspace(0, parameters['sr'] / 2, 5).astype(int).tolist()
                    xl = np.round(np.linspace(0.0, sample_length, 5), 2).tolist()
                    ax.set_xlabel('Time (s)')
                    ax.set(xticks=xl, xticklabels=xl)
                    ax.set_ylabel('Frequency (Hz)')
                    ax.set(yticks=yl, yticklabels=yl)
                    ax.set_title(f"{onset // parameters['sr']}s - {offset // parameters['sr']}s")
                    plt.savefig(os.path.join(images_dir, f'{base_file_name}_sample_{i}.png'), bbox_inches='tight', dpi=100)
                    plt.close()

                    df.to_csv(os.path.join(exported_results_dir, f'{base_file_name}_labels.tsv'), sep='\t')

                    bar_audio.update(1)

            bar_audio.close()
            bar.update(1)

    print('Done.')
