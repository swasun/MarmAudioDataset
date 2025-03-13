# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import utils as u
from tqdm import tqdm
import models
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, default='detections_1910_fixed_positions.pkl')
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--dataloader_n_jobs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='resnet18_logMel128_oct2020split.stdc')
    parser.add_argument('--species_name', type=str, default='marmoset')
    args = parser.parse_args()

    df = pd.read_pickle(os.path.join(args.root_experiment_folder, args.detection_pkl_file_name))
    df = df[(~df.type.isna())]
    df = df[(~df.onset.isna())]
    traindf, testdf = train_test_split(df, stratify=df.AEtype, random_state=42, test_size=0.2)
    traindf = df

    nepoch = 100
    batch_size = 32
    writer = SummaryWriter(f'{args.root_experiment_folder}/runs/'+ args.model_name)
    print('Go for model '+ args.model_name)
    lr = 0.005
    wdL2 = 0.002

    if args.species_name == 'marmoset':
        num_classes = len(u.idxtotype_marmoset)
        labels = u.marmoset_labels
    elif args.species_name == 'voc_vs_noise':
        num_classes = len(u.idxtotype_voc_vs_noise)
        labels = u.voc_vs_noise_labels
    print(num_classes)
    model = torch.nn.Sequential(models.get[f'frontend_logMel_{args.sampling_rate_khz}'], models.get_resnet50(num_classes=num_classes))
    print(model)

    gpu = torch.device(args.device)
    model.to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=wdL2, lr=lr, momentum=.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.95 ** epoch)
    trainLoader = torch.utils.data.DataLoader(u.Dataset(traindf, args.audio_folder, brown=True, sampleDur=.5, retType=True, species_name=args.species_name), batch_size=batch_size, shuffle=True, num_workers=args.dataloader_n_jobs, prefetch_factor=4)
    testLoader = torch.utils.data.DataLoader(u.Dataset(testdf, args.audio_folder, sampleDur=0.5, retType=True, species_name=args.species_name), batch_size=batch_size, shuffle=True, num_workers=args.dataloader_n_jobs)
    loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([np.clip(1000/(df.type==label).sum(), 1, 50) for label in labels]))

    step = 0
    for epoch in range(nepoch):
        model.train()
        for x, name, label in tqdm(trainLoader, desc=str(epoch), leave=False):
            x = x.to(gpu)
            optimizer.zero_grad()
            pred = model(x).cpu().squeeze()
            score = loss(pred, label)
            score.backward()
            optimizer.step()
            writer.add_scalar('loss', score.item(), step)
            writer.add_scalar('train acc', balanced_accuracy_score(label, pred.argmax(-1)), step)
            writer.add_scalar('train F1', f1_score(label.detach(), pred.argmax(-1).detach(), average='macro'), step)
            step += 1

        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        model.eval()
        preds, labels = [], []
        for batch in tqdm(testLoader, desc=str(epoch), leave=False):
            x, name, label = batch
            x = x.to(gpu)
            pred = model(x).cpu().detach().squeeze()
            preds.extend(pred.numpy())
            labels.extend(label)

        preds = np.array(preds)
        labels = np.array(labels)

        testloss = loss(torch.Tensor(preds), torch.Tensor(labels).long())
        scheduler.step()

        writer.add_scalar('test loss', testloss, epoch)
        writer.add_scalar('test acc', balanced_accuracy_score(labels, preds.argmax(-1)), epoch)
        writer.add_scalar('test F1', f1_score(labels, preds.argmax(-1), average='macro'), epoch)

        torch.save(model.state_dict(), os.path.join(args.root_experiment_folder, args.model_name))
