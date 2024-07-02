import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import torch
from tqdm import tqdm
from sklearn import metrics
import models, utils as u
import argparse, os


def _test(df, name, model, threshold=None):
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(u.Dataset(df, args.audio_folder, sampleDur=.5, species_name=args.species_name, retType=True), \
            batch_size=batch_size, shuffle=True, num_workers=args.dataloader_n_jobs, collate_fn=u.collate_fn)
        labels, preds, names, logits = [], [], [], []
        for x, idx, label in tqdm(loader):
            x = x.to(gpu, non_blocking=True)
            pred = model(x).cpu().detach().view(len(x), num_classes).numpy()
            preds.extend(pred.argmax(-1))
            labels.extend(label.numpy())
            logits.extend(pred)
            names.extend(idx)
        labels, preds, logits, names = np.array(labels), np.array(preds), np.array(logits), np.array(names)
        print(logits.shape)
        print(name+' : '+str(len(df))+' samples')
        print(df.type.value_counts())
        acc = (labels==preds).sum() / len(labels)
        fails[name+'_acc'] = acc
        print('Accuracy : '+str(acc))
        top3acc = metrics.top_k_accuracy_score(labels, logits, k=3, labels=np.arange(num_classes))
        print('Top 3 acc', top3acc)
        fails[name+'_top3acc'] = top3acc
        f1 = metrics.f1_score(labels, preds, average='weighted')
        fails[name+'_f1'] = acc
        print(f'Balanced acc: {metrics.balanced_accuracy_score(labels, preds)}')
        print('F1 score : '+str(f1))
        confusion_matrix = metrics.confusion_matrix(labels, preds, labels=np.arange(num_classes), normalize='true')
        print(confusion_matrix)

        fails[name+'_failnames'] = names[(preds!=labels)]
        fails[name+'_failpreds'] = preds[(preds!=labels)]
        metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=u.typetoidx_marmoset.keys()).plot()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name}.png')
        plt.close()


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
    traindf = df[~((df.year==2020)&(df.month==10))]
    testdf = df[((df.year==2020)&(df.month==10))]

    print(traindf.type.value_counts())
    print(testdf.type.value_counts())

    if args.species_name == 'marmoset':
        num_classes = len(u.idxtotype_marmoset)
    batch_size = 64

    model = torch.nn.Sequential(models.get[f'frontend_logMel_{args.sampling_rate_khz}'], models.get_resnet18(num_classes=num_classes))
    model.load_state_dict(torch.load(os.path.join(args.root_experiment_folder, args.model_name)))
    gpu = torch.device('cuda')
    model.eval().to(gpu)

    print('Performance of :'+args.model_name)
    fails = {}
    _test(traindf, 'train', model)
    _test(testdf, 'test', model)
    np.save('testreport_'+args.model_name.rsplit('.', 1)[0], fails)
