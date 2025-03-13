# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from tqdm import tqdm
import models
import os
import torch
import utils as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import joblib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Print vocalization spectrograms sorted by a classifier\'s predictions to verify its quality on unannotated samples.")
    parser.add_argument('idx', type=int)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, required=True)
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--dataloader_n_jobs', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--encodings_file_name', type=str, default='encodings_AE_marmosset_logMel128_16feat_all-vocs.npy')
    parser.add_argument('--species_name', type=str, default='marmoset')
    args = parser.parse_args()

    frontend = models.get[f'frontend_logMel_{args.sampling_rate_khz}']
    gpu = torch.device(args.device)
    frontend.to(gpu)

    df = pd.read_pickle(f'{args.root_experiment_folder}/{args.detection_pkl_file_name}')
    print(len(df))

    dic = joblib.load(f'{args.root_experiment_folder}/{args.encodings_file_name}')
    encodings = dic['encodings']
    X = dic['umap']
    idxs = pd.Series(dic['idx'])

    encodings = encodings[idxs.isin(df.index)]
    X = X[idxs.isin(df.index)]
    idxs = idxs[idxs.isin(df.index)]

    center = encodings[idxs==args.idx][0]
    os.system(f'mkdir {args.root_experiment_folder}/close_to_{args.idx}')

    dists = np.sum(np.square(encodings-center), axis=1)
    closest = idxs[dists < np.quantile(dists, 0.001)]
    loader = torch.utils.data.DataLoader(u.Dataset(df.loc[closest], args.audio_folder, species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs, shuffle=True, pin_memory=True)
    for batch in tqdm(loader):
        x, idx = batch
        x = x.to(gpu)
        x = frontend(x).cpu().detach().squeeze()
        plt.imshow(x, origin='lower', aspect='auto')
        plt.savefig(f'{args.root_experiment_folder}/close_to_{args.idx}/'+str(idx.squeeze().item()))
        plt.close()
