# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

from vocalseg.utils import butter_bandpass_filter

import argparse
import noisereduce as nr
import librosa
import os
from scipy.io.wavfile import write
import glob2
from tqdm import tqdm
import pathlib


"""
python src/denoise_wavfile.py --noise_file_path=data/MarmosetVocalizations2019/marmoset_room_noise.wav --input_folder_path=data/MarmosetVocalizations2019/chunked/2019_12_29_2 --output_folder_path=data/MarmosetVocalizations2019/denoised/2019_12_29_2
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--noise_file_path', nargs='?', type=str, required=True)
    parser.add_argument('--input_folder_path', nargs='?', type=str, default=None)
    parser.add_argument('--input_file_path', nargs='?', type=str, default=None)
    parser.add_argument('--output_folder_path', nargs='?', type=str, default='denoised')
    parser.add_argument('--prop_decrease', nargs='?', type=float, default=0.6)
    parser.add_argument('--n_std_thresh', nargs='?', type=float, default=2.0)
    parser.add_argument('--sr', nargs='?', type=int, default=96000)
    args = parser.parse_args()

    if (not args.input_folder_path and not args.input_file_path) or \
        (args.input_folder_path and args.input_file_path):
        raise ValueError('Use either input_folder_path or input_file_path')

    noise, sr = librosa.load(args.noise_file_path, sr=args.sr)
    base_noise_file_name = os.path.splitext(os.path.basename(args.noise_file_path))[0]

    if args.input_folder_path:
        print(args.input_folder_path)
        file_paths = glob2.glob(os.path.join(args.input_folder_path, '**/*.wav'))
    else:
        file_paths = [args.input_file_path]

    with tqdm(file_paths) as bar:
        for file_path in bar:
            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            bar.set_description(base_file_name)
            output_path = os.path.join(args.output_folder_path, base_file_name)
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

            base_file_name = os.path.splitext(os.path.basename(file_path))[0]
            bar.set_description(base_file_name)
            audio, sr = librosa.load(file_path, sr=args.sr)
            output_file_name = f"{base_file_name}_denoised_n-std-thresh-{args.n_std_thresh}_prop-decrease-{args.prop_decrease}.wav"
            output_file_path = os.path.join(args.output_folder_path, base_file_name, output_file_name)
            if os.path.isfile(output_file_path):
                print(f'{output_file_path} already exists...')
                continue
            print(f'Computing {output_file_path}...')
            audio = butter_bandpass_filter(
                audio, 200, 8999, args.sr, order=5
            )
            denoised_audio = nr.reduce_noise(audio_clip=audio, noise_clip=noise,
                prop_decrease=args.prop_decrease, n_std_thresh=args.n_std_thresh)
            write(output_file_path, args.sr, denoised_audio)
            bar.update(1)
