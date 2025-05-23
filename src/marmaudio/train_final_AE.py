# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import os
import torch
import glob
from torchvision.utils import make_grid
from torch import nn, device, optim, save, isnan
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import models
import utils as u
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random
import shutil


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fig


"""
python train_final_AE.py MarmAudioData/Vocalizations --root_experiment_folder=MarmAudioDataset/experiment_results --dataloader_n_jobs=7
python train_final_AE.py Vocalizations --root_experiment_folder=experiment_results --dataloader_n_jobs=4
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='experiment_results')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--sampling_rate_khz', type=int, default=96)
    parser.add_argument('--sample_duration', type=float, default=0.5)
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='AE_marmoset_logMel128_256feat_all-segmented-vocs_fixrelu')
    parser.add_argument('--frontend', type=str, default='logMel')
    parser.add_argument('--species_name', type=str, default='marmoset')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--reset_logs', action='store_true', default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.experiment_name is None:
        experiment_name = f'{args.model_name}_{datetime.now().strftime("%d-%m-%Y-%H-%M")}'
    else:
        experiment_name = args.experiment_name
    output_folder = os.path.join(args.root_experiment_folder, experiment_name)
    checkpoint_folder = os.path.join(output_folder, 'checkpoint')
    logs_folder = f'{args.root_experiment_folder}/{experiment_name}/logs'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)

    if args.reset_logs:
        shutil.rmtree(logs_folder, ignore_errors=True)

    file_paths = glob.glob(os.path.join(args.audio_folder, '*', '*'))
    parent_names, file_names, labels = list(), list(), list()
    for file_path in tqdm(file_paths):
        parent_name = os.path.split(os.path.dirname(file_path))[-1]
        file_name = os.path.basename(file_path)
        label = os.path.splitext(os.path.basename(file_path))[0].split('_')[:-1][0]

        parent_names.append(parent_name)
        file_names.append(file_name)
        labels.append(label)
    df = pd.DataFrame().from_dict({'parent_name': parent_names, 'file_name': file_names, 'label': labels})
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.label.value_counts())

    marmoset_labels = df.label.unique()
    typetoidx_marmoset = {k:i for i, k in enumerate(marmoset_labels)}

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df[['label']])

    nepoch = 300
    batch_size = 64
    nMel = 128
    nfeat = 256
    device = device(args.device)
    lr = 0.003
    wdL2 = 0.0
    updates_per_epoch_train = 2000
    total_samples_per_epoch_train = batch_size * updates_per_epoch_train
    updates_per_epoch_test = 64
    total_samples_per_epoch_test = batch_size * updates_per_epoch_test
    writer = SummaryWriter(logs_folder)
    vgg16 = u.VGG()
    vgg16.eval().to(device)
    save_every_epoch = 50

    if args.sampling_rate_khz == 96:
        sr = 96000
    elif args.sampling_rate_khz == 44_1:
        sr = 44100
    else:
        raise ValueError(f'Invalid sr {args.sampling_rate_khz}')

    print('Experiment '+experiment_name)

    frontend = models.frontend['logMel'](sr, 1024, args.sample_duration, nMel)
    encoder = models.__dict__['sparrow_encoder'](*((nfeat // 16, (4, 4)) if nMel == nMel else (nfeat // 8, (2, 4))))
    decoder = models.sparrow_decoder(nfeat, (4, 4) if nMel == 128 else (2, 4))
    model = torch.nn.Sequential(frontend, encoder, decoder).to(device)

    optimizer = optim.AdamW(model.parameters(), weight_decay=wdL2, lr=lr, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : .99**epoch)

    if args.experiment_name is not None:
        first_epoch = 100
        print(f"Loading already existing {os.path.join(checkpoint_folder, f'model_{first_epoch}.stdc')}...")
        model.load_state_dict(torch.load(os.path.join(checkpoint_folder, f'model_{first_epoch}.stdc'), map_location=device))
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_folder, f'optimizer_{first_epoch}.stdc'), map_location=device))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_folder, f'scheduler_{first_epoch}.stdc'), map_location=device))
    else:
        first_epoch = 1
    
    trainLoader = torch.utils.data.DataLoader(u.FinalDataset(train_df, args.audio_folder, typetoidx_marmoset=typetoidx_marmoset, brown=False, sampleDur=.5, retType=True, species_name='marmoset'), batch_size=batch_size, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=8)
    testLoader = torch.utils.data.DataLoader(u.FinalDataset(test_df, args.audio_folder, typetoidx_marmoset=typetoidx_marmoset, brown=False, sampleDur=.5, retType=True, species_name='marmoset'), batch_size=batch_size, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=8)

    loss_fun = nn.MSELoss()

    step = 0
    for epoch in range(first_epoch, nepoch+1):
        # Randomly sample indices for the subset
        dataset_size = len(trainLoader.dataset)
        sampled_indices = random.sample(range(dataset_size), total_samples_per_epoch_train)
        
        # Create a subset of the dataset with the sampled indices
        subset = torch.utils.data.Subset(trainLoader.dataset, sampled_indices)
        
        # Create a new DataLoader for the subset
        subset_train_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle since indices are already randomized
            num_workers=args.dataloader_n_jobs,
            prefetch_factor=8,
        )

        model.train()
        for x, name, label in tqdm(subset_train_loader, desc=str(epoch), leave=False):
            optimizer.zero_grad()
            label = frontend(x.to(device))
            assert not torch.isnan(label).any(), "NaN in spectrogram :'( "+str(name[torch.isnan(label).any(1).any(1).any(1)])
            x = encoder(label)
            pred = decoder(x)
            predd = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
            labell = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

            score = loss_fun(predd, labell)
            score.backward()
            optimizer.step()
            writer.add_scalar('train_loss', score.item(), step)

            if step%50==0 :
                images = [(e-e.min())/(e.max()-e.min()) for e in label[:8]]
                original_grid = make_grid(images)
                writer.add_image('train_target', original_grid, step)
                writer.add_embedding(x.detach(), global_step=step, label_img=label)
                images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
                reconstructed_grid = make_grid(images)
                writer.add_image('train_reconstruct', reconstructed_grid, step)

            step += 1

        if epoch % 10 == 0:
            scheduler.step()

        print('[INFO] Evaluation...')
        model.eval()

        with torch.no_grad():
            # Randomly sample indices for the subset
            dataset_size = len(testLoader.dataset)
            sampled_indices = random.sample(range(dataset_size), total_samples_per_epoch_test)
            
            # Create a subset of the dataset with the sampled indices
            subset = torch.utils.data.Subset(testLoader.dataset, sampled_indices)
            
            # Create a new DataLoader for the subset
            subset_test_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,  # No need to shuffle since indices are already randomized
                num_workers=args.dataloader_n_jobs,
                prefetch_factor=8,
            )

            test_scores = list()
            for x, name, label in tqdm(subset_test_loader, desc=str(epoch), leave=False):
                label = frontend(x.to(device))
                assert not isnan(label).any(), 'Found a NaN in input spectrogram'
                pred = decoder(encoder(label))
                predd = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
                labell = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))
                score = loss_fun(predd, labell)
                test_scores.append(score.item())
            writer.add_scalar('test_loss', np.mean(test_scores), epoch)
            images = [(e-e.min())/(e.max()-e.min()) for e in label[:8]]
            original_grid = make_grid(images)
            writer.add_image('test_target', original_grid, epoch)
            images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
            reconstructed_grid = make_grid(images)
            writer.add_image('test_reconstruct', reconstructed_grid, epoch)
        if epoch % save_every_epoch == 0:
            save(model.state_dict(), os.path.join(checkpoint_folder, f'model_{epoch}.stdc'))
            save(optimizer.state_dict(), os.path.join(checkpoint_folder, f'optimizer_{epoch}.stdc'))
            save(scheduler.state_dict(), os.path.join(checkpoint_folder, f'scheduler_{epoch}.stdc'))
        else:
            save(model.state_dict(), os.path.join(checkpoint_folder, f'model_last.stdc'))
            save(optimizer.state_dict(), os.path.join(checkpoint_folder, f'optimizer_last.stdc'))
            save(scheduler.state_dict(), os.path.join(checkpoint_folder, f'scheduler_last.stdc'))
