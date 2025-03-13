# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import pandas as pd
import argparse
import os
from tqdm import tqdm
import pathlib


def retreive_peaks_labels(file_path):
    # Read the data
    df = pd.read_csv(file_path, sep='\t')

    # Add an 'offset_diff' column to store the difference between consecutive offsets
    df['offset_diff'] = df['offset'].diff()

    # Find where the group changes (either label or offset_diff > time_threshold)
    time_threshold = 0.5  # Adjust this value based on the desired closeness
    group_change = (df['pred_label'].ne(df['pred_label'].shift())) | (df['offset_diff'] > time_threshold)

    # Assign a group ID for each group
    df['group_id'] = group_change.cumsum()

    # Compute max_conf_points and first/last offsets for each group
    aggregated = df.groupby('group_id').agg(
        max_conf_point=('pred_conf_SM', 'idxmax'),
        pred_first_offset=('offset', 'first'),
        pred_last_offset=('offset', 'last')
    )

    # Extract the rows corresponding to the max_conf_points and merge with the aggregated data
    new_df = df.loc[aggregated['max_conf_point']].merge(aggregated, left_index=True, right_on='max_conf_point')
    
    # Drop the temporary columns and reset the index
    new_df.drop(columns=['offset_diff', 'group_id'], inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    return new_df


"""
python retreive_peaks_from_predictions.py ../../../../Marmoset/uncut_raw predictions_lensample0.5_stride0.1_thresh0.0
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_folder', type=str, help='Path of the folder with audio files to process')
    parser.add_argument('prediction_folder', type=str, default='')
    args = parser.parse_args()

    file_paths = [os.path.join(args.prediction_folder, 'predictions_' + file_path.replace('.wav', f"_{args.prediction_folder.replace('predictions_', '')}.csv")) for file_path in os.listdir(args.audio_folder)]
    output_folder = f'filtered_{args.prediction_folder}'
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    for source_file_path in tqdm(file_paths):
        if not os.path.isfile(source_file_path):
            continue
        destination_file_path = os.path.join(output_folder, os.path.basename(source_file_path))
        if os.path.isfile(destination_file_path):
            continue
        df = retreive_peaks_labels(source_file_path).reset_index(drop=True)
        df.to_csv(destination_file_path, sep='\t')
