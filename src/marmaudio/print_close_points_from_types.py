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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, required=True)
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--vocalization_types', nargs='+', default=['Tsik-Ek', 'Scream', 'Seep', 'Ek', 'Trill-Phee'])
    parser.add_argument('--dataloader_n_jobs', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--encodings_file_name', type=str, default='encodings_AE_marmosset_logMel128_16feat_all-vocs.npy')
    parser.add_argument('--species_name', type=str, default='marmoset')
    args = parser.parse_args()

    os.system(f'mkdir {args.root_experiment_folder}/close_pngs/')

    frontend = models.get[f'frontend_logMel_{args.sampling_rate_khz}']
    gpu = torch.device(args.device)
    frontend.to(gpu)

    df = pd.read_pickle(f'{args.root_experiment_folder}/{args.detection_pkl_file_name}')
    df = df[df.type.isin(args.vocalization_types)]

    dic = np.load(f'{args.root_experiment_folder}/{args.encodings_file_name}', allow_pickle=True).item()
    encodings = dic['encodings']
    X = dic['umap']
    idxs = pd.Series(dic['idx'])

    encodings = encodings[idxs.isin(df.index)]
    X = X[idxs.isin(df.index)]
    idxs = idxs[idxs.isin(df.index)]

    for type, grp in df.groupby('type'):
        mask = idxs.isin(grp.index)
        center = np.median(X[mask], axis=0)
        dists = np.sum(np.square(X-center), axis=1)
        closest = idxs[~mask][dists[~mask] < np.quantile(dists[~mask], 0.005)]
        print(type, closest.sum())
        loader = torch.utils.data.DataLoader(u.Dataset(df.loc[closest], args.audio_folder, species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs, shuffle=True, pin_memory=True)
        print(type, len(grp))
        os.system(f'mkdir {args.root_experiment_folder}/close_pngs/'+type)
        for batch in tqdm(loader, desc=type):
            x, idx = batch
            x = x.to(gpu)
            x = frontend(x).cpu().detach().squeeze()
            plt.imshow(x, origin='lower', aspect='auto')
            #plt.xticks(np.arange(0, 128, 20), np.arange(0, 128, 20)/128)
            plt.savefig(f'{args.root_experiment_folder}/close_pngs/'+type+'/'+str(idx.squeeze().item()))
            plt.close()
