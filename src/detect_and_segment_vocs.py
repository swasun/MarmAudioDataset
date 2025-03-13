# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from vocalseg.utils import butter_bandpass_filter, spectrogram, plot_spec
from vocalseg.continuity_filtering import continuity_segmentation

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pathlib
from scipy.io import wavfile
import argparse
import pandas as pd
from tqdm import tqdm
import json
import glob2
import sys
import gc
import soundfile as sf


"""
python src/detect_and_segment_vocs.py --input_folder_path=data/MarmosetVocalizations2019/denoised --experiment_name=parameters_v21-06-21 --parameter_file=experiments/marmoset_parameters_v21-06-21.json --output_folder_path=results/MarmosetVocalizations2019
python src/detect_and_segment_vocs.py --input_folder_path=data/MarmosetVocalizations2019/denoised --experiment_name=parameters_v21-06-21 --parameter_file=experiments/marmoset_parameters_v21-06-21.json --output_folder_path=results/MarmosetVocalizations2019 --exception_files 2019_12_27_denoised_n-std-thresh-2.0_prop-decrease-0.6 2019_12_29_3_0_denoised_n-std-thresh-2.0_prop-decrease-0.6 2019_12_29_3_1_denoised_n-std-thresh-2.0_prop-decrease-0.6 2019_12_29_denoised_n-std-thresh-2.0_prop-decrease-0.6
python src/detect_and_segment_vocs.py --input_folder_path=data/MarmosetVocalizations2019/denoised/2019_12_29_3 --experiment_name=parameters_v21-06-21 --parameter_file=experiments/marmoset_parameters_v21-06-21.json --output_folder_path=results/MarmosetVocalizations2019/2019_12_29_3 --chunk_folder_path=data/MarmosetVocalizations2019/chunked/2019_12_29_3 --write_samples
python src/detect_and_segment_vocs.py --input_folder_path=data/MarmosetVocalizations2020/denoised/2020_03_05 --experiment_name=parameters_v21-06-21 --parameter_file=experiments/marmoset_parameters_v21-06-21.json --output_folder_path=results/MarmosetVocalizations2020/2020_03_05 --chunk_folder_path=data/MarmosetVocalizations2020/chunked/2020_03_05 --write_samples
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_folder_path', nargs='?', type=str, default=None)
    parser.add_argument('--input_file_path', nargs='?', type=str, default=None)
    parser.add_argument('--chunk_folder_path', nargs='?', type=str, default=None)
    parser.add_argument('--exception_files', nargs='+', type=str, default=None)
    parser.add_argument('--experiment_name', nargs='?', type=str, required=True, help='Name of the experiment.')
    parser.add_argument('--parameter_file', nargs='?', type=str, default=None, help='Parameter file path to override the default parameters without using argparse.')
    parser.add_argument('--sr', nargs='?', type=int, default=96000, help='Specify the sampling args.sr.')
    parser.add_argument('--min_level_db', nargs='?', type=int, default=-80, help='Default dB minimum of spectrogram (threshold anything below).')
    parser.add_argument('--min_level_db_floor', nargs='?', type=int, default=-40, help='Highest number min_level_db is allowed to reach dynamically.')
    parser.add_argument('--db_delta', nargs='?', type=int, default=5, help='Delta in setting min_level_db.')
    parser.add_argument('--n_fft', nargs='?', type=int, default=1024, help='FFT window size.')
    parser.add_argument('--hop_length_ms', nargs='?', type=int, default=1, help='Number audio of frames in ms between STFT columns.')
    parser.add_argument('--win_length_ms', nargs='?', type=int, default=5, help='Size of fft window (ms).')
    parser.add_argument('--ref_level_db', nargs='?', type=int, default=20, help='Reference level dB of audio.')
    parser.add_argument('--pre', nargs='?', type=float, default=0.97, help='Coefficient for preemphasis filter.')
    parser.add_argument('--spectral_range', nargs='+', type=int, default=[125, 48000], help='Spectral range to care about for spectrogram.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Display output.')
    parser.add_argument('--write_samples', action='store_true', default=False, help='Write the samples and the spectrograms.')
    parser.add_argument('--load_results', action='store_true', default=False, help='Try to load the existing results instead of running the segmentation.')
    parser.add_argument('--mask_thresh_std', nargs='?', type=int, default=1, help='Standard deviations above median to threshold out noise (higher = threshold more noise).')
    parser.add_argument('--neighborhood_time_ms', nargs='?', type=int, default=5, help='Size in time of neighborhood-continuity filter.')
    parser.add_argument('--neighborhood_freq_hz', nargs='?', type=int, default=500, help='Size in Hz of neighborhood-continuity filter.')
    parser.add_argument('--neighborhood_thresh', nargs='?', type=float, default=0.5, help='Threshold number of neighborhood time-frequency bins above 0 to consider a bin not noise.')
    parser.add_argument('--min_syllable_length_s', nargs='?', type=float, default=0.1, help='Shortest expected length of syllable.')
    parser.add_argument('--min_silence_for_spec', nargs='?', type=float, default=0.1, help='Shortest expected length of silence in a song (used to set dynamic threshold).')
    parser.add_argument('--silence_threshold', nargs='?', type=float, default=0.05, help='Threshold for spectrogram to consider noise as silence.')
    parser.add_argument('--max_vocal_for_spec', nargs='?', type=float, default=1.0, help='Longest expected vocalization in seconds.')
    parser.add_argument('--temporal_neighbor_merge_distance_ms', nargs='?', type=float, default=0.0, help='Longest distance at which two elements should be considered one.')
    parser.add_argument('--overlapping_element_merge_thresh', nargs='?', type=float, default=np.inf, help='Proportion of temporal overlap to consider two elements one.')
    parser.add_argument('--min_element_size_ms_hz', nargs='+', type=int, default=[0, 0], help='Smallest expected element size (in ms and Hz). Everything smaller is removed.')
    parser.add_argument('--figsize', nargs='?', type=tuple, default=(30, 3), help='Size of figure for displaying output.')
    parser.add_argument('--lowpass_filter', nargs='?', type=int, default=125, help='Low value of the butter bandpass filter (in Hz).')
    parser.add_argument('--output_folder_path', nargs='?', type=str, default='results', help='Root path of the output folder.')
    parser.add_argument('--minimal_duration', nargs='?', type=float, default=1.0, help='The minimal duration of the saved samples in sec.')
    parser.add_argument('--print_stacktrace', action='store_true', default=False, help='Print the stacktrace when an error is encountered.')
    parser.add_argument('--override', action='store_true', default=False, help='Override the already existing result files.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode that will save interim plots.')
    args = parser.parse_args()

    if (not args.input_folder_path and not args.input_file_path) or \
        (args.input_folder_path and args.input_file_path):
        raise ValueError('Use either input_folder_path or input_file_path')

    parameters = args.__dict__

    if args.parameter_file:
        if not os.path.isfile(args.parameter_file):
            raise ValueError(f"Parameter file not found '{args.parameter_file}'")
        with open(args.parameter_file) as f:
            loaded_parameters = json.load(f)
        for key in loaded_parameters:
            if loaded_parameters[key] != parameters[key]:
                print(f"Overriding parameter '{key}' from {parameters[key]} -> {loaded_parameters[key]}", file=sys.stderr)
                parameters[key] = loaded_parameters[key]

    if args.input_folder_path:
        file_paths = glob2.glob(os.path.join(parameters['input_folder_path'], '**/*.wav'))
    else:
        file_paths = [parameters['input_file_path']]

    pathlib.Path(os.path.join(parameters['output_folder_path'], parameters['experiment_name'])).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(parameters['output_folder_path'], parameters['experiment_name'], f"{parameters['experiment_name']}_parameters.txt"), 'w') as f:
        json.dump(parameters, f, indent=4)

    with tqdm(file_paths) as bar:
        for file_path in bar:
            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            base_file_name = base_file_name.replace('_denoised_n-std-thresh-2.0_prop-decrease-0.8', '') if args.chunk_folder_path else base_file_name
            if args.exception_files and base_file_name in args.exception_files:
                bar.set_description(f'Skipping file {base_file_name}...')
                bar.update(1)
                continue
            else:
                bar.set_description(base_file_name)

            print('Creating the folder tree...', file=sys.stderr)
            exported_results_dir = os.path.join(parameters['output_folder_path'], parameters['experiment_name'], base_file_name)
            audios_dir = os.path.join(exported_results_dir, 'audios')
            images_dir = os.path.join(exported_results_dir, 'images')
            logging_dir = os.path.join(exported_results_dir, 'logging')
            pathlib.Path(audios_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(images_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(logging_dir).mkdir(parents=True, exist_ok=True)

            if os.path.isfile(os.path.join(exported_results_dir, f'{base_file_name}_results.pickle')) and \
                os.path.isfile(os.path.join(exported_results_dir, f'{base_file_name}_labels.tsv')):
                bar.set_description(f'Results already exists for {base_file_name}. Skipping...')
                print(f'Results already exists for {base_file_name}. Skipping...', file=sys.stderr)
                bar.update(1)
                continue

            print(f"Loading wavfile {file_path}...", file=sys.stderr)
            _, data = wavfile.read(file_path)

            if os.path.isfile(os.path.join(exported_results_dir, f'{base_file_name}_results.pickle')) and not args.override:
                with open(os.path.join(exported_results_dir, f'{base_file_name}_results.pickle'), 'rb') as f:
                    results = pickle.load(f)
            elif not args.load_results:
                print('Computing the spectrogram...', file=sys.stderr)
                spec = spectrogram(
                    data,
                    parameters['sr'],
                    n_fft=parameters['n_fft'],
                    hop_length_ms=parameters['hop_length_ms'],
                    win_length_ms=parameters['win_length_ms'],
                    ref_level_db=parameters['ref_level_db'],
                    pre=parameters['pre'],
                    min_level_db=parameters['min_level_db'],
                )

                print('Saving the spectrogram plot...', file=sys.stderr)
                fig, ax = plt.subplots(figsize=parameters['figsize'])
                plot_spec(spec, fig, ax)
                fig.savefig(os.path.join(logging_dir, f'{base_file_name}_spect_filtered.png'))

                print('Running the continuity segmentation algorithm...', file=sys.stderr)

                results = continuity_segmentation(
                    data,
                    parameters['sr'],
                    n_fft=parameters['n_fft'],
                    hop_length_ms=parameters['hop_length_ms'],
                    win_length_ms=parameters['win_length_ms'],
                    ref_level_db=parameters['ref_level_db'],
                    pre=parameters['pre'],
                    min_level_db=parameters['min_level_db'],
                    verbose=True if args.debug else parameters['verbose'],
                    silence_threshold=parameters['silence_threshold'],
                    spectral_range=parameters['spectral_range'],
                    mask_thresh_std=parameters['mask_thresh_std'],
                    figsize=parameters['figsize'],
                    min_silence_for_spec=parameters['min_silence_for_spec'],
                    neighborhood_thresh=parameters['neighborhood_thresh'],
                    neighborhood_time_ms=parameters['neighborhood_time_ms'],
                    neighborhood_freq_hz=parameters['neighborhood_freq_hz'],
                    temporal_neighbor_merge_distance_ms=parameters['temporal_neighbor_merge_distance_ms'],
                    overlapping_element_merge_thresh=parameters['overlapping_element_merge_thresh'],
                    min_element_size_ms_hz=parameters['min_element_size_ms_hz'],
                    max_vocal_for_spec=parameters['max_vocal_for_spec'],
                    interim_plot_id=f'{logging_dir}/{base_file_name}' if args.debug else None)

                print('Exporting the results...', file=sys.stderr)
                with open(os.path.join(exported_results_dir, f'{base_file_name}_results.pickle'), 'wb') as f:
                    pickle.dump(results, f)
            else:
                with open(os.path.join(exported_results_dir, f'{base_file_name}_results.pickle'), 'rb') as f:
                    results = pickle.load(f)

            try:
                figsize = (100, 4)
                from matplotlib.patches import Rectangle
                from matplotlib.collections import PatchCollection
                fig, axs = plt.subplots(nrows = 2, figsize=(figsize[0], figsize[1]*2))

                plot_spec(results['spec'], fig, axs[0], parameters['sr'], hop_len_ms=parameters['hop_length_ms'], show_cbar=False)
                axs[1].plot(results['vocal_envelope'])
                axs[1].set_xlim([0, len(results['vocal_envelope'])])

                ylmin, ylmax = (axs[0].get_ylim())
                ysize = (ylmax - ylmin)*.1
                ymin = ylmax- ysize

                patches = []
                for onset, offset in zip(results['onsets'], results['offsets']):
                    axs[0].axvline(onset, color = '#FFFFFF', ls="dashed", lw=0.5)
                    axs[0].axvline(offset, color = '#FFFFFF', ls="dashed", lw=0.5)
                    patches.append(Rectangle(xy=(onset, ymin), width = offset-onset, height = ysize))

                collection = PatchCollection(patches, color='white', alpha=0.5)
                axs[0].add_collection(collection)
                plt.savefig(os.path.join(logging_dir, f'{base_file_name}_labelled_spec_patches.png'))
                plt.close()
            except Exception as e:
                print(f'[ERROR] [{file_path}] error while plotting {base_file_name}_labelled_spec_patches.png')
                if args.print_stacktrace:
                    print(e)
                continue

            minimal_frame_number = int(parameters['minimal_duration'] * parameters['sr'])

            if args.chunk_folder_path:
                print(args.chunk_folder_path)
                print(file_path)
                print(os.path.join(args.chunk_folder_path, os.path.basename(file_path)))
                chunk_folder_path = os.path.dirname(args.chunk_folder_path) if os.path.isfile(args.chunk_folder_path) else args.chunk_folder_path
                file_path = os.path.join(chunk_folder_path, os.path.basename(file_path)).replace('_denoised_n-std-thresh-2.0_prop-decrease-0.8', '')
                print(f"Loading wavfile {file_path}...", file=sys.stderr)
                _, data = wavfile.read(file_path)
                print(f'Filtering data using a bandpass filter...', file=sys.stderr)
                # TODO add parameter
                #data = butter_bandpass_filter(data, parameters['lowpass_filter'], (parameters['sr']//2)-1, parameters['sr'], order=2)
                data = butter_bandpass_filter(data, 200, 8999, parameters['sr'], order=5)

            j = 0
            onset_series, offset_series, id_series = list(), list(), list()
            for i in range(len(results['onsets'])):
                onset = int(results['onsets'][i] * parameters['sr'])
                offset = int(results['offsets'][i] * parameters['sr'])
                additional_frame_number = int(0.2 * parameters['sr'])
                if (offset - onset) < additional_frame_number:
                    onset -= (additional_frame_number // 2 )
                    offset += (additional_frame_number // 2)
                if args.write_samples:
                    print(f'Writing sample #{i}', file=sys.stderr)
                    sample = data[onset:offset]
                    sf.write(os.path.join(audios_dir, f'{base_file_name}_sample_{j}.wav'), sample, parameters['sr'])

                    try:
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
                    except:
                        print(f'Error with spectrogram generation of sample #{i}')
                        continue

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
                    plt.savefig(os.path.join(images_dir, f'{base_file_name}_sample_{j}.png'), bbox_inches='tight', dpi=100)
                    plt.close()

                    del spec
                    gc.collect()

                onset_series.append(onset)
                offset_series.append(offset)
                id_series.append(j)
                j += 1

            df = pd.DataFrame.from_dict({'onset': onset_series, 'offset': offset_series, 'id': id_series})
            df.to_csv(os.path.join(exported_results_dir, f'{base_file_name}_labels.tsv'), sep='\t')

            del data
            del results
            gc.collect()

            bar.update(1)

    print('Done.', file=sys.stderr)
