import os
from torchvision.utils import make_grid
from torch import nn, utils, device, optim, save, isnan
import pandas as pd
import utils as u
from tqdm import tqdm
from models import get
from torch.utils.tensorboard import SummaryWriter
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, default='detections_1910_fixed_positions.pkl')
    parser.add_argument('--sampling_rate_khz', type=int, default=96)
    parser.add_argument('--sample_duration', type=float, default=0.5)
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='AE_marmosset_logMel128_256feat_all-vocs.stdc')
    parser.add_argument('--frontend', type=str, default='logMel')
    parser.add_argument('--species_name', type=str, default='marmoset')

    args = parser.parse_args()

    df = pd.read_csv(args.detection_pkl_file_name, sep='\t')
    print(f'Training using {len(df)} vocalizations')

    nepoch = 300
    batch_size = 64
    nfeat = 256
    device = device(args.device)
    lr = 0.003
    wdL2 = 0.0
    writer = SummaryWriter(f'{args.root_experiment_folder}/runs/'+args.model_name)
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
    loader = utils.data.DataLoader(u.Dataset(df, args.audio_folder, sampleDur=args.sample_duration, species_name=args.species_name, from_raw_recordings=False), \
                                   batch_size=batch_size, shuffle=True, num_workers=args.dataloader_n_jobs, collate_fn=u.collate_fn)
    loss_fun = nn.MSELoss()

    step = 0
    for epoch in range(nepoch):
        for x, name in tqdm(loader, desc=str(epoch), leave=False):
            optimizer.zero_grad()
            label = frontend(x.to(device))
            if isnan(label).any():
                print()
                print(label.shape)
                import numpy as np
                nan_count = np.sum(np.isnan(label.detach().cpu().numpy()))
                print(nan_count)
                exit()
            assert not isnan(label).any(), 'Found a NaN in input spectrogram'
            x = encoder(label)
            pred = decoder(x)
            predd = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
            labell = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

            score = loss_fun(predd, labell)
            score.backward()
            optimizer.step()
            writer.add_scalar('loss', score.item(), step)

            if step%50==0 :
                images = [(e-e.min())/(e.max()-e.min()) for e in label[:8]]
                grid = make_grid(images)
                writer.add_image('target', grid, step)
                writer.add_embedding(x.detach(), global_step=step, label_img=label)
                images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
                grid = make_grid(images)
                writer.add_image('reconstruct', grid, step)

            step += 1
        if epoch % 10 == 0:
            scheduler.step()
        save(model.state_dict(), os.path.join(args.root_experiment_folder, args.model_name))
