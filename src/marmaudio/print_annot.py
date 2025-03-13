# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import matplotlib.pyplot as plt
import models
import utils as u
from tqdm import tqdm
import os
import torch
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, default='detections_1910_fixed_positions.pkl')
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    parser.add_argument('--output_folder', type=str, default='annot_pngs')
    parser.add_argument('--species_name', type=str, default='marmoset')
    args = parser.parse_args()

    df = pd.read_pickle(os.path.join(args.root_experiment_folder, args.detection_pkl_file_name))
    #df = df[((~df.type.isin(['Vocalization', 'Noise', 'mixed']))&(~df.type.isna())&(~df.type.str.contains('-').astype(bool)))]
    #df = df[((~df.type.isin(['Vocalization', 'Noise', 'mixed']))&(~df.type.isna()))]
    df = df[df.type == 'Noise']

    gpu = torch.device(args.device)
    frontend = models.get[f'frontend_logMel_{args.sampling_rate_khz}'].to(gpu)

    if os.path.isdir(f'{args.root_experiment_folder}/{args.output_folder}'):
        os.system(f'rm -R {args.root_experiment_folder}/{args.output_folder}/*')
    else:
        os.system(f'mkdir {args.root_experiment_folder}/{args.output_folder}')

    with torch.no_grad():
        for type, grp in df.groupby('type'):
            type = type.replace(' ','_')
            os.system(f'mkdir {args.root_experiment_folder}/{args.output_folder}/{type}')
            #for x, idx in tqdm(torch.utils.data.DataLoader(u.Dataset(grp.sample(200) if len(grp) > 200 else grp, args.audio_folder, sampleDur=.5, species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs), leave=False, desc=str(type)):
            for x, idx in tqdm(torch.utils.data.DataLoader(u.Dataset(grp, args.audio_folder, sampleDur=.5, species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs), leave=False, desc=str(type)):
                x = x.to(gpu)
                x = frontend(x).cpu().detach().squeeze()
                plt.imshow(x, origin='lower', aspect='auto')
                plt.savefig(f'{args.root_experiment_folder}/{args.output_folder}/{type}/{idx.squeeze().item()}')
                plt.close()
