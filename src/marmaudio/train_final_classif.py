import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import utils as u
from tqdm import tqdm
import models
import argparse
import os
import glob


"""
python train_final_classif.py /media/charly/9A0CEE3F0CEE1653/Charly/Vocalizations --root_experiment_folder=/media/SSD2/Charly/MonkeyVocalizations/experiment_results --subset=train --model_name=resnet50_logMel128_trainsplit_wo_brown_wo_vocs --dataloader_n_jobs=7
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--dataloader_n_jobs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

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
    df = df[df.label != 'Vocalization']
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.label.value_counts())

    marmoset_labels = df.label.unique()
    typetoidx_marmoset = {k:i for i, k in enumerate(marmoset_labels)}

    output_folder = os.path.join(args.root_experiment_folder, args.model_name)
    os.makedirs(output_folder, exist_ok=True)

    df.to_csv(os.path.join(output_folder, f'{args.model_name}_labels_used.tsv'), sep='\t')

    if args.subset == 'train':
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df[['label']])
        print(train_df.label.value_counts())
        print(test_df.label.value_counts())
        train_df.to_csv(os.path.join(output_folder, f'{args.model_name}_train_labels_used.tsv'), sep='\t')
        test_df.to_csv(os.path.join(output_folder, f'{args.model_name}_test_labels_used.tsv'), sep='\t')
    elif args.subset == 'all':
        train_df = df
        print(train_df.label.value_counts())
        train_df.to_csv(os.path.join(output_folder, f'{args.model_name}_train_labels_used.tsv'), sep='\t')
    else:
        raise ValueError('Invalid subset')

    sampling_rate_khz = '96'
    nepoch = 20
    batch_size = 64
    print('Go for model '+ args.model_name)
    lr = 0.005
    wdL2 = 0.002

    labels = df.label.unique()
    num_classes = len(labels)
    print(num_classes)
    model = torch.nn.Sequential(models.get[f'frontend_logMel_{sampling_rate_khz}'], models.get_resnet50(num_classes=num_classes))
    print(model)

    gpu = torch.device(args.device)
    model.to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=wdL2, lr=lr, momentum=.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.95 ** epoch)
    trainLoader = torch.utils.data.DataLoader(u.FinalDataset(train_df, args.audio_folder, typetoidx_marmoset=typetoidx_marmoset, brown=False, sampleDur=.5, retType=True, species_name='marmoset'), batch_size=batch_size, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=4)
    testLoader = torch.utils.data.DataLoader(u.FinalDataset(test_df, args.audio_folder, typetoidx_marmoset=typetoidx_marmoset, brown=False, sampleDur=.5, retType=True, species_name='marmoset'), batch_size=batch_size, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=4)
    loss = torch.nn.CrossEntropyLoss()

    step = 0
    for epoch in range(nepoch):
        model.train()
        train_losses, train_accs = list(), list()
        for x, name, label in tqdm(trainLoader, desc=str(epoch+1), leave=False):
            x = x.to(gpu)
            label = label.to(gpu)
            optimizer.zero_grad()
            pred = model(x)
            score = loss(pred, label)
            score.backward()
            optimizer.step()
            current_score = score.item()
            train_losses.append(current_score)
            train_acc = balanced_accuracy_score(label.detach().cpu(), pred.detach().cpu().argmax(-1), adjusted=True)
            train_accs.append(train_acc)
            step += 1
            if step % 1000 == 0:
                print(f'[Epoch {epoch+1}] Train loss: {current_score} - Train acc: {train_acc}')

        if args.subset != 'all':
            model.eval()
            test_losses, test_accs = list(), list()
            with torch.no_grad():
                for x, name, label in tqdm(testLoader, desc=str(epoch+1), leave=False):
                    x = x.to(gpu)
                    label = label.to(gpu)
                    pred = model(x)
                    score = loss(pred, label)
                    test_losses.append(score.item())
                    test_accs.append(balanced_accuracy_score(label.detach().cpu(), pred.detach().cpu().squeeze().argmax(-1), adjusted=True))
            print(f'[Epoch {epoch+1}] Test loss: {np.mean(test_losses)} - Test acc: {np.mean(test_accs)}')

        torch.save(model.state_dict(), os.path.join(output_folder, f'{args.model_name}_epoch{epoch+1}.stdc'))
        train_history_df = pd.DataFrame().from_dict({'train_loss': train_losses, 'train_acc': train_accs})
        train_history_df['epoch'] = epoch+1
        train_history_df.to_csv(os.path.join(output_folder, f'{args.model_name}_epoch{epoch+1}_train_history.tsv'), sep='\t')
        if args.subset != 'all':
            test_history_df = pd.DataFrame().from_dict({'test_loss': test_losses, 'test_acc': test_accs})
            test_history_df['epoch'] = epoch+1
            test_history_df.to_csv(os.path.join(output_folder, f'{args.model_name}_epoch{epoch+1}_test_history.tsv'), sep='\t')
