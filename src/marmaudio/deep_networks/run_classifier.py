# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import matplotlib.pyplot as plt
import os
import torch
import models
import utils as u
from scipy import signal
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run this script to use a CNN for the classification of marmosset vocalisations on a folder of audio files.")
    parser.add_argument('audio_folder', type=str, help='Path of the folder with audio files to process')
    parser.add_argument('stdc', type=str, help='Filename of the weight file (.stdc)')
    parser.add_argument('-lensample', type=float, help='Length of the signal for each sample (in seconds)', default=.5)
    parser.add_argument('-stride', type=float, help='Size of the stride between each prediction', default=0.1)
    parser.add_argument('-batch_size', type=int, help='Amount of samples to process at a time (usefull for parallel computation using a GPU)', default=32)
    parser.add_argument('-channel', type=int, help='Channel of the audio file to use in the model inference (starting from 0)', default=0)
    parser.add_argument('-print', help='Print high predictions into pred_pngs', action='store_true')
    parser.add_argument('-threshold', help='Confidence threshold to keep a prediction', type=float, default=0.8)
    parser.add_argument('-output_filename', type=str, help='Name of the output file for saving predictions', default='')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--species_name', type=str, default='marmoset')
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    parser.add_argument('--audio_file_id', type=int, default=None)
    parser.add_argument('--output_folder_name', type=str, default='')
    parser.add_argument('--file_names_to_exclude', nargs='+', type=str, default=[])
    parser.set_defaults(maxPool=True)
    args = parser.parse_args()

    if args.output_folder_name != '':
        pathlib.Path(args.root_experiment_folder, args.output_folder_name).mkdir(parents=True, exist_ok=True)

    def collate_fn(batch):
        batch = [item for item in batch if item is not None and item[0].numel() > 0]
        if len(batch) == 0:
            return None

        data, meta = zip(*batch)
        max_length = max([x.shape[0] for x in data])
        padded_data = [torch.cat((x, torch.zeros(max_length - x.shape[0])), dim=0) for x in data]
        stacked_data = torch.stack(padded_data, dim=0)

        return stacked_data, meta

    norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)
    if args.sampling_rate_khz == '96':
        model_fs = 96_000
    elif args.sampling_rate_khz == '44_1':
        model_fs = 44_100
    else:
        raise ValueError('Unsupported sampling rate')

    if args.species_name == 'marmoset':
        typetoidx = u.typetoidx_marmoset
        idxtotype = u.idxtotype_marmoset
        labels = u.marmoset_labels
    elif args.species_name == 'voc_vs_noise':
        typetoidx = u.typetoidx_voc_vs_noise
        idxtotype = u.idxtotype_voc_vs_noise
        labels = u.voc_vs_noise_labels

    # Pytorch dataset class to load audio samples
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            super(Dataset, self)
            self.samples = []
            file_paths = os.listdir(args.audio_folder)
            new_file_paths = list()
            for file_path in file_paths:
                full_file_name = f"predictions_{os.path.splitext(file_path)[0]}_lensample{args.lensample}_stride{args.stride}_thresh{args.threshold}.csv"
                if not os.path.isfile(os.path.join(args.root_experiment_folder, args.output_folder_name, full_file_name)):
                    new_file_paths.append(file_path)
            file_paths = list(set(new_file_paths) - set(args.file_names_to_exclude))
            if args.audio_file_id is not None:
                file_paths.sort()
                file_paths = [file_paths[args.audio_file_id]]
            for fn in tqdm(file_paths, desc='Dataset initialization', leave=False):
                try:
                    info = sf.info(os.path.join(args.audio_folder, fn))
                    duration, fs = info.duration, info.samplerate
                    self.samples.extend([{'fn':fn, 'offset':offset, 'fs':fs} for offset in np.arange(0, duration+.01 - args.lensample, args.stride)])
                except:
                    continue
                assert info.channels > args.channel, f"The desired channel is unavailable for the audio file {fn}"
            print(f'Loaded {len(self.samples)} samples to process')

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            try:
                sig, fs = sf.read(os.path.join(args.audio_folder, sample['fn']),
                                  start=int(sample['offset']*sample['fs']),
                                  stop=int(sample['offset']*sample['fs']) + int(args.lensample*sample['fs']),
                                  always_2d=True)
            except:
                print('Failed loading '+sample['fn'])
                return None
            sig = sig[:, args.channel]
            if fs != model_fs:
                sig = signal.resample(sig, int(args.lensample * model_fs))
            sig = norm(sig)
            return torch.tensor(sig).float(), sample

    # prepare model
    model = torch.nn.Sequential(models.get[f'frontend_logMel_{args.sampling_rate_khz}'], models.get_resnet50(len(labels)))
    model.load_state_dict(torch.load(os.path.join(args.root_experiment_folder, args.stdc)))
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # prepare data loader and output storage for predictions
    loader = torch.utils.data.DataLoader(Dataset(), batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.dataloader_n_jobs)
    if len(loader) == 0:
        print(f'Unable to open any audio file in the given folder {args.audio_folder}')
        exit()

    out = []
    # forward the model on each batch
    with torch.inference_mode():
        for x, meta in tqdm(loader, desc='Model inference'):
            x = x.to(device)
            pred = model(x).cpu().detach().view(len(x), -1)
            pred_SM = torch.nn.functional.softmax(pred, dim=-1)
            values, indices = torch.max(pred_SM, axis=-1)

            filtered_indices = np.where(values > args.threshold)[0]
            
            filtered_predictions = [{'fn': meta[i]['fn'],
                                    'offset': meta[i]['offset'],
                                    'pred_label': idxtotype[indices[i].item()],
                                    'pred_conf_SM': values[i].item(),
                                    'pred_conf': pred[i, indices[i]].item()} for i in filtered_indices if idxtotype[indices[i].item()] != 'Noise']

            out.extend(filtered_predictions)

            if args.print:
                for i in np.where(values > args.threshold)[0]:
                    if idxtotype[indices[i].item()] != 'Noise':
                        pathlib.Path(f'{args.root_experiment_folder}/{args.output_folder_name}/pred_pngs/{idxtotype[indices[i].item()]}').mkdir(parents=True, exist_ok=True)
                        plt.imshow(model[0](x[i].unsqueeze(0)).squeeze().cpu().numpy(), origin='lower', aspect='auto')
                        plt.savefig(f'{args.root_experiment_folder}/{args.output_folder_name}/pred_pngs/{idxtotype[indices[i].item()]}/{meta[i]["fn"]}_{meta[i]["offset"].item():.1f}.png')
                        plt.close()

    out = pd.DataFrame().from_dict(out)
    if args.audio_file_id is not None:
        pred_fn = f"predictions_{os.path.splitext(out['fn'].unique()[0])[0]}_lensample{args.lensample}_stride{args.stride}_thresh{args.threshold}"
        if args.maxPool:
            pred_fn += '.csv'
        else:
            pred_fn += '.pkl'
        print(f'Saving results into {args.root_experiment_folder}/{args.output_folder_name}/{pred_fn}')
        if pred_fn.endswith('csv'):
            out.to_csv(os.path.join(args.root_experiment_folder, args.output_folder_name, pred_fn), index=False, sep='\t')
        else:
            out.to_pickle(os.path.join(args.root_experiment_folder, args.output_folder_name, pred_fn))
    else:
        dfs = dict(tuple(out.groupby('fn')))
        for fn, df in dfs.items():
            pred_fn = f"predictions_{os.path.splitext(fn)[0]}_lensample{args.lensample}_stride{args.stride}_thresh{args.threshold}"
            if args.maxPool:
                pred_fn += '.csv'
            else:
                pred_fn += '.pkl'
            
            print(f'Saving results into {args.root_experiment_folder}/{args.output_folder_name}/{pred_fn}')
            if pred_fn.endswith('csv'):
                df.to_csv(os.path.join(args.root_experiment_folder, args.output_folder_name, pred_fn), index=False, sep='\t')  # Use df instead of out
            else:
                df.to_pickle(os.path.join(args.root_experiment_folder, args.output_folder_name, pred_fn))  # Use df instead of out
