# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import argparse
import librosa
from scipy.io.wavfile import write
import pathlib
import os
import numpy as np
import glob2
from tqdm import tqdm


"""
python src/chunk_wavfile.py --input_folder_path=data/MarmosetVocalizations2019/raw --output_folder_path=data/MarmosetVocalizations2019/chunked
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_folder_path', nargs='?', type=str, default=None)
    parser.add_argument('--input_file_path', nargs='?', type=str, default=None)
    parser.add_argument('--output_folder_path', nargs='?', type=str, required=True)
    parser.add_argument('--sr', nargs='?', type=int, default=96000)
    parser.add_argument('--chunk_duration', nargs='?', type=int, default=10, help='In minute')
    args = parser.parse_args()

    if (not args.input_folder_path and not args.input_file_path) or \
        (args.input_folder_path and args.input_file_path):
        raise ValueError('Use either input_folder_path or input_file_path')

    if args.input_folder_path:
        file_paths = glob2.glob(os.path.join(args.input_folder_path, '**/*.wav'))
    else:
        file_paths = [args.input_file_path]

    chunk_len = 60 * args.chunk_duration * args.sr

    with tqdm(file_paths) as bar:
        for file_path in bar:
            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            bar.set_description(base_file_name)

            output_path = os.path.join(args.output_folder_path, base_file_name)
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
            audio, sr = librosa.load(file_path, sr=args.sr)

            if len(audio) < chunk_len:
                print(f'Audio len already lower than {args.chunk_duration}min. Exiting.')
                if os.path.isfile(os.path.join(output_path, f'{base_file_name}.wav')):
                    continue
                write(os.path.join(output_path, f'{base_file_name}.wav'),
                    args.sr, audio)
                bar.update(1)
                continue

            n_chunks = len(audio) // chunk_len
            chunks = np.array_split(audio, n_chunks + 1)

            for i in range(len(chunks)):
                if os.path.isfile(os.path.join(output_path, f'{base_file_name}_{i}.wav')):
                    print(f"{os.path.join(output_path, f'{base_file_name}_{i}.wav')} already exists...")
                    continue
                print(f'Writing chunk #{i+1}...')
                write(os.path.join(output_path, f'{base_file_name}_{i}.wav'),
                    args.sr, chunks[i])

            bar.update(1)
