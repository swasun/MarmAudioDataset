# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

from marmaudio.classifier import load_classifier, run_classifier, prediction_to_str

import soundfile as sf
import glob
import pandas as pd
import os


if __name__ == "__main__":
    marmoset_labels = pd.read_csv('resnet50_logMel128_trainsplit_wo_brown_wo_vocs_labels_used.tsv', sep='\t').label.unique()
    typetoidx_marmoset = {k:i for i, k in enumerate(marmoset_labels)}
    idxtotype_marmoset = {typetoidx_marmoset[k]:k for k in typetoidx_marmoset}

    clf = load_classifier('resnet50_logMel128_trainsplit_wo_brown_wo_vocs_epoch1.stdc', marmoset_labels)

    file_paths = glob.glob('Vocalizations/*/*.wav')[0:10]
    annotations = pd.read_csv('Annotations.tsv', sep='\t') # Read the annotations
    file_names = [os.path.basename(file_path) for file_path in file_paths]
    file_names = [file_name.replace('.wav', '.flac') for file_name in file_names]
    true_labels = annotations[annotations.file_name.isin(file_names)].label.tolist()

    predicted_labels = list()
    for i, file_path in enumerate(file_paths):
        signal, sampling_rate = sf.read(file_path) # Read the vocalization waveform and store it as 'signal'
        predictions = run_classifier(clf, signal)
        predicted_label = prediction_to_str(predictions, idxtotype_marmoset)
        predicted_labels.append(predicted_label)
        print(f'[{file_path}]  True: {true_labels[i]}  Predicted: {predicted_label}')
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    accuracy = (correct_predictions / len(true_labels)) * 100
    print(f'Accuracy: {accuracy}%')
