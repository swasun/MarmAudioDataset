# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from marmaudio.filterbank import STFT, MelFilter, Log1p

import torch
from torch import nn
import torchvision
import numpy as np


def get_resnet18(num_classes=11):
    resnet = torchvision.models.resnet18()
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet = nn.Sequential(*[k for k in resnet.children()]) # turn resnet into a sequential instead of named modules
    resnet[-2] = nn.Conv2d(512, num_classes, (4, 1)) # Size 4 works for 128 frequency bins
    resnet[-1] = nn.AdaptiveMaxPool2d(output_size=(1,1))
    return resnet

def get_resnet50(num_classes=11):
    resnet = torchvision.models.resnet50()
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.fc = nn.Linear(2048, num_classes)
    return resnet

def load_classifier(checkpoint_path, marmoset_labels):
    frontend = nn.Sequential(
        STFT(1024, 368),
        MelFilter(96000, 1024, 128, 1000, 48000),
        Log1p(7))
    model = torch.nn.Sequential(frontend, get_resnet50(len(marmoset_labels)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model = model.to(device)
    return model

def run_classifier(clf, signal):
    norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)
    sig = signal[np.newaxis, ...]
    sig = norm(sig)
    predictions = clf(torch.tensor(sig).cpu().float())
    return predictions

def prediction_to_str(predictions, idxtotype_marmoset):
    pred_SM = torch.nn.functional.softmax(predictions, dim=-1)
    _, indices = torch.max(pred_SM, axis=-1)
    indices = indices.detach().squeeze().cpu().numpy()
    prediction_str = idxtotype_marmoset[indices.item()]
    return prediction_str
