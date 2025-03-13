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
from models import get
import utils as u
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


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
python train_final_AE.py /media/charly/9A0CEE3F0CEE1653/Charly/Vocalizations --root_experiment_folder=/media/SSD2/Charly/MarmAudioDataset/experiment_results --dataloader_n_jobs=7
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--sampling_rate_khz', type=int, default=96)
    parser.add_argument('--sample_duration', type=float, default=0.5)
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='AE_marmoset_logMel128_256feat_all-segmented-vocs.stdc')
    parser.add_argument('--frontend', type=str, default='logMel')
    parser.add_argument('--species_name', type=str, default='marmoset')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    experiment_name = f'{args.model_name}_{datetime.now().strftime("%d-%m-%Y-%H-%M")}'
    output_folder = os.path.join(args.root_experiment_folder, experiment_name)
    os.makedirs(output_folder, exist_ok=True)

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

    nepoch = 100
    batch_size = 32
    nfeat = 256
    device = device(args.device)
    lr = 0.003
    wdL2 = 0.0
    writer = SummaryWriter(f'{args.root_experiment_folder}/{experiment_name}/logs')
    vgg16 = u.VGG()
    vgg16.eval().to(device)

    if args.sampling_rate_khz == 96:
        sr = 96000
    elif args.sampling_rate_khz == 44_1:
        sr = 44100
    else:
        raise ValueError(f'Invalid sr {args.sampling_rate_khz}')

    frontend = get[f'frontend'](sr, 1024, args.sample_duration, 128).eval()
    encoder = get['sparrow_encoder'](nfeat // 16, (4, 4))
    decoder = get['sparrow_decoder'](nfeat, (4, 4))
    model = nn.Sequential(frontend, encoder, decoder).to(device)

    print('Go for model '+args.model_name)

    optimizer = optim.AdamW(model.parameters(), weight_decay=wdL2, lr=lr, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : .99**epoch)
    
    trainLoader = torch.utils.data.DataLoader(u.FinalDataset(train_df, args.audio_folder, typetoidx_marmoset=typetoidx_marmoset, brown=False, sampleDur=.5, retType=True, species_name='marmoset'), batch_size=batch_size, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=4)
    testLoader = torch.utils.data.DataLoader(u.FinalDataset(test_df, args.audio_folder, typetoidx_marmoset=typetoidx_marmoset, brown=False, sampleDur=.5, retType=True, species_name='marmoset'), batch_size=batch_size, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=4)

    loss_fun = nn.MSELoss()

    step = 0
    for epoch in range(nepoch):
        model.train()
        for x, name, label in tqdm(trainLoader, desc=str(epoch), leave=False):
            optimizer.zero_grad()
            label = frontend(x.to(device))
            assert not isnan(label).any(), 'Found a NaN in input spectrogram'
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
                #show(original_grid).savefig(os.path.join(output_folder, f'original_grid_step{step}.png'))
                writer.add_image('train_target', original_grid, step)
                writer.add_embedding(x.detach(), global_step=step, label_img=label)
                images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
                reconstructed_grid = make_grid(images)
                #show(reconstructed_grid).savefig(os.path.join(output_folder, f'reconstructed_grid_step{step}.png'))
                writer.add_image('train_reconstruct', reconstructed_grid, step)
                break

            step += 1

        if epoch % 10 == 0:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            for x, name, label in tqdm(testLoader, desc=str(epoch), leave=False):
                label = frontend(x.to(device))
                assert not isnan(label).any(), 'Found a NaN in input spectrogram'
                x = encoder(label)
                pred = decoder(x)
                predd = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
                labell = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

                images = [(e-e.min())/(e.max()-e.min()) for e in label[:8]]
                original_grid = make_grid(images)
                writer.add_image('test_target', original_grid, step)

                images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
                reconstructed_grid = make_grid(images)
                writer.add_image('test_reconstruct', reconstructed_grid, step)

                score = loss_fun(predd, labell)
                writer.add_scalar('test_loss', score.item(), step)
        save(model.state_dict(), os.path.join(args.root_experiment_folder, args.model_name))
