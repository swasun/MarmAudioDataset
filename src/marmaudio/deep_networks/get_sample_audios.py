# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import pandas as pd
import argparse
import soundfile as sf
import os
from tqdm import tqdm
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, default='detections.pkl')
    parser.add_argument('--sampling_rate', type=int, default=96000)
    parser.add_argument('--sample_duration', type=float, default=3.0)
    args = parser.parse_args()

    df = pd.read_pickle(os.path.join(args.root_experiment_folder, args.detection_pkl_file_name))
    df = df[df.type == 'Noise']
    

    pathlib.Path(os.path.join(args.root_experiment_folder, 'audios')).mkdir(parents=True, exist_ok=True)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        sampleDur = args.sampling_rate * args.sample_duration
        sig, fs = sf.read(f'{args.audio_folder}/{row.parent_name}.wav', start=int(row.pos - sampleDur//2), stop=int(row.pos + sampleDur//2))
        sig = sig[:,0]
        sf.write(f'{args.root_experiment_folder}/audios/{index}.wav', sig, args.sampling_rate)
