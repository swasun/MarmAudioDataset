# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import models
import utils as u
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Print vocalization spectrograms sorted by a classifier\'s predictions to verify its quality on unannotated samples.")
    parser.add_argument('modelname', type=str, help='Filename of the classifier\'s weights (.stdc)')
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, required=True)
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--species_name', type=str, default='marmoset')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--predict', type=int, default=1)
    parser.add_argument('--dataloader_n_jobs', type=int, default=4)
    args = parser.parse_args()

    df = pd.read_pickle(f'{args.root_experiment_folder}/{args.detection_pkl_file_name}')
    df = df[df.type.isna()]

    if os.path.isdir(f'{args.root_experiment_folder}/pred_pngs'):
        os.system(f'rm -R {args.root_experiment_folder}/pred_pngs/*')
    else:
        os.system(f'mkdir {args.root_experiment_folder}/pred_pngs')

    if args.species_name == 'marmoset':
        typetoidx = u.typetoidx_marmoset
        idxtotype = u.idxtotype_marmoset
        labels = u.marmoset_labels
    elif args.species_name == 'voc_vs_noise':
        typetoidx = u.typetoidx_voc_vs_noise
        idxtotype = u.idxtotype_voc_vs_noise
        labels = u.voc_vs_noise_labels

    for type in typetoidx:
        os.system(f"mkdir {args.root_experiment_folder}/pred_pngs/{type.replace(' ','_')}")

    model = torch.nn.Sequential(models.get[f'frontend_logMel_{args.sampling_rate_khz}'], models.get_resnet50(len(labels)))
    model.load_state_dict(torch.load(os.path.join(args.root_experiment_folder, args.modelname)))
    model.eval().to(args.device)

    loader = torch.utils.data.DataLoader(u.Dataset(df, args.audio_folder, sampleDur=.5, species_name=args.species_name), batch_size=128, num_workers=args.dataloader_n_jobs)
    preds, conf, idxs = [], [], []
    with torch.inference_mode():
        for x, idx in tqdm(loader, desc='get preds'):
            pred = model(x.to(args.device)).detach().cpu().view(len(x), -1)
            conf, pred = torch.nn.functional.softmax(pred, dim=-1).max(-1)
            df.loc[idx, 'pred_label'] = [idxtotype[p.item()] for p in pred]
            df.loc[idx, 'pred_conf'] = conf.numpy()

    df.to_pickle(f'{args.root_experiment_folder}/{args.detection_pkl_file_name}_all_preds')

    print(f'#vocs found: {len(df)}')
    df = df[df.pred_conf>df.pred_conf.quantile(.75)].reset_index(drop=True)
    print(f'{len(df)} vocs left after filtereing by confidence: {df.pred_conf.quantile(.75)}')

    for p, grp in df.groupby('pred_label'):
        loader = torch.utils.data.DataLoader(u.Dataset(grp, args.audio_folder, species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs, prefetch_factor=4)
        for x, idx in tqdm(loader, desc=p):
            x = model[0](x.to(args.device))
            plt.figure()
            plt.imshow(x.cpu().squeeze().detach(), origin='lower', aspect='auto')
            plt.savefig(f"{args.root_experiment_folder}/pred_pngs/{p.replace(' ','_')}/{idx.item()}")
            plt.close()
