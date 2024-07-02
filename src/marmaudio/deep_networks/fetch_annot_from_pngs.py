# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import pandas as pd
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, \
        description="""This script fetches the annotations stored via the sorting of .png files.
                    For each sample (.png file), the name of its parent folder will be set as label in the detection.pkl file (column type).
                    For instance, create a folder named annotation_toto. In it, create a folder Phee containing .png spectrograms of Phee vocalisations,
                    and another folder named Trill etc...
                    In these folders, add samples of the correct type to store annotations.""")
    parser.add_argument('annot_folder', type=str, help='Name of the folder containing annotations.')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_pickle(os.path.join(args.root_experiment_folder, args.detection_pkl_file_name))

    for label in os.listdir(args.annot_folder):
        for file in os.listdir(f'{args.annot_folder}/{label}'):
            df.loc[int(file.split('.')[0]), 'type'] = label

    df.to_pickle(os.path.join(args.root_experiment_folder, args.detection_pkl_file_name))
