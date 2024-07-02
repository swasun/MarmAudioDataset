# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import os
import argparse
import pandas as pd
from tqdm import tqdm
import utils as u
import numpy as np
import soundfile as sf
from scipy import signal
import models
import torch
import matplotlib.pyplot as plt
import pathlib


"""
python cut_predicted_audios.py ../../../../Marmoset/maybe_cut_raw/raw --lensample=2 --root_experiment_folder=Marmoset_voc19-20-21_16-03-23 --labels_filename=prediction_labels_16-03-23.csv
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run this script to use a CNN for the classification of marmosset vocalisations on a folder of audio files.")
    parser.add_argument('audio_folder', type=str, help='Path of the folder with audio files to process')
    parser.add_argument('--lensample', type=float, help='Length of the signal for each sample (in seconds)', default=5),
    parser.add_argument('--labels_filename', type=str, help='Name of the output file for saving predictions', default='')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--species_name', type=str, default='marmoset')
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('-channel', type=int, help='Channel of the audio file to use in the model inference (starting from 0)', default=0)
    args = parser.parse_args()

    if args.species_name == 'marmoset':
        typetoidx = u.typetoidx_marmoset
        idxtotype = u.idxtotype_marmoset
        labels = u.marmoset_labels
    elif args.species_name == 'voc_vs_noise':
        typetoidx = u.typetoidx_voc_vs_noise
        idxtotype = u.idxtotype_voc_vs_noise
        labels = u.voc_vs_noise_labels

    if args.sampling_rate_khz == '96':
        model_fs = 96_000 
    elif args.sampling_rate_khz == '44_1':
        model_fs = 44_100
    else:
        raise ValueError('Unsupported sampling rate')

    norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)
    df = pd.read_csv(os.path.join(args.root_experiment_folder, args.labels_filename), sep='\t')
    for i in range(len(labels)):
        df[f'prediction_{i}'] = df[f'prediction_{i}'].astype(float)

    model = torch.nn.Sequential(models.get[f'frontend_logMel_{args.sampling_rate_khz}']).to('cuda:0')

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        filename = row['filename']
        offset = row['offset']
        row.drop(['filename', 'offset'], inplace=True)
        if max([row[f'prediction_{i}'] for i in range(len(labels))]) <= 0.9:
            continue
        label = idxtotype[np.argmax([row[f'prediction_{i}'] for i in range(len(labels))])]
        if label == 'Noise':
            continue
        
        info = sf.info(os.path.join(args.audio_folder, filename))
        duration, fs = info.duration, info.samplerate
        sig, fs = sf.read(os.path.join(args.audio_folder, filename), start=int(offset*fs), stop=int((offset+args.lensample)*fs), always_2d=True)
        sig = sig[:, args.channel]
        if fs != model_fs:
            sig = signal.resample(sig, int(args.lensample * model_fs))
        sig = norm(sig)
        pathlib.Path(f"{args.root_experiment_folder}/{label}").mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.imshow(model(torch.tensor(sig).to('cuda:0').float().unsqueeze(0))[0].detach().cpu().numpy().squeeze(), origin='lower', aspect='auto')
        plt.savefig(f"{args.root_experiment_folder}/{label}/{index}.png")
        plt.close()
