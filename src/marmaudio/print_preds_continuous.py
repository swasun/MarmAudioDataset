# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from scipy import signal
from tqdm import tqdm
import os, argparse
import matplotlib.pyplot as plt
import pandas as pd
import models, utils as u
import torch
import soundfile as sf


"""
python print_preds_continuous.py ../../../../Marmoset/maybe_cut_raw/raw/2020_01_07_2.wav resnet18_logMel128_oct2020split.stdc --root_experiment_folder=Marmoset_voc19-20-21_16-03-23 --dataloader_n_jobs=8
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Print vocalization spectrograms sorted by a classifier\'s predictions to verify its quality on unannotated samples.")
    parser.add_argument('audio_file', type=str, help='Path to the file to process')
    parser.add_argument('modelname', type=str, help='Filename of the classifier\'s weights (.stdc)')
    parser.add_argument('--root_experiment_folder', type=str, default='./')
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--sample_dur', type=int, default='60')
    parser.add_argument('--species_name', type=str, default='marmoset')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataloader_n_jobs', type=int, default=4)
    args = parser.parse_args()

    f = sf.SoundFile(args.audio_file)

    if args.species_name == 'marmoset':
        typetoidx = u.typetoidx_marmoset
        idxtotype = u.idxtotype_marmoset
        labels = u.marmoset_labels
    elif args.species_name == 'voc_vs_noise':
        typetoidx = u.typetoidx_voc_vs_noise
        idxtotype = u.idxtotype_voc_vs_noise
        labels = u.voc_vs_noise_labels

    for type in labels:
        os.system(f"mkdir -p {args.root_experiment_folder}/pred_pngs/{type.replace(' ','_')}")

    model = torch.nn.Sequential(models.get[f'frontend_logMel_{args.sampling_rate_khz}'], models.get_resnet18(len(labels))[:-1])
    model.load_state_dict(torch.load(os.path.join(args.root_experiment_folder, args.modelname)))
    model.eval().to(args.device)
    loader = torch.utils.data.DataLoader(u.Continuous_Dataset(args.audio_file, args.sample_dur, species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs)

    pos_to_sec = lambda pos : ((pos * 32) * 368 + 512) / 96000 # ( (pos * stride) * hopsize + winsize/2 )/ fs

    out = []
    with torch.no_grad():
        for x, offset in tqdm(loader, desc='get preds'):
            pred = model(x.to(args.device)).detach().cpu().squeeze()
            conf, _ = torch.nn.functional.softmax(pred, dim=-1).max(-1)
            conf = conf.detach().cpu().numpy()
            pred = torch.nn.functional.softmax(pred, dim=-2).numpy()
            for i, label in enumerate(labels):
                peaks = signal.find_peaks(pred[i], height=0.98)[0]
                out.extend([{'pos': offset.item() + pos_to_sec(p), 'pred_label':label, 'pred_conf':conf[i]} for p in peaks])

    df = pd.DataFrame().from_dict(out)
    df['parent_name'] = args.audio_file[:-4]
    df.pos = (df.pos * 96000).astype(int)

    print(f'#vocs found: {len(df)}')
    df = df[df.pred_conf>df.pred_conf.quantile(.75)].reset_index(drop=True)
    print(f'#vocs found after filtereing by confidence: {len(df)}')
    df.to_pickle(f"{args.root_experiment_folder}/{df.parent_name.unique()[0].split('/')[-1]}_labels.pkl")

    for p, grp in df.groupby('pred_label'):
        loader = torch.utils.data.DataLoader(u.Dataset(grp, './', species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs, prefetch_factor=4)
        for x, idx in tqdm(loader, desc=p):
            x = model[0](x.to(args.device))
            plt.figure()
            plt.imshow(x.cpu().squeeze().detach(), origin='lower', aspect='auto')
            row = df.loc[idx.item()]
            plt.savefig(f"{args.root_experiment_folder}/pred_pngs/{p.replace(' ','_')}/{row.parent_name.split('/')[-1]}_{row.pos}")
            plt.close()
