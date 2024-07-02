# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import os
import shutil
import pandas as pd
import math
from tqdm.contrib.concurrent import thread_map
from threading import Lock
import collections
import glob


def copy_file(file_info):
    (index, file_id, prediction_type, year, month, folder_counters, src_base_path, dest_base_path, max_files_per_folder, lock) = file_info
    prediction_type = 'Infant_cry' if prediction_type == 'Infant cry' else prediction_type
    src_file = os.path.join(src_base_path, prediction_type, f"{file_id}.wav")
    
    if not os.path.exists(src_file):
        print(f'{src_file} not found')
        return

    with lock:
        year_month_key = f"{year}_{month}"
        folder_id = folder_counters.get(year_month_key, 0) // max_files_per_folder
        folder_counters[year_month_key] += 1

    year_month_folder = f"{year_month_key}_{folder_id}"
    dest_folder = os.path.join(dest_base_path, year_month_folder)
    dest_file = os.path.join(dest_folder, f"{prediction_type}_{index}.wav")

    if os.path.isfile(dest_file):
        return

    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy2(src_file, dest_file)


if __name__ == "__main__":
    segmented_predictions_id = 'marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-09-08'
    tsv_file = f"marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13_fixed2.tsv"
    src_base_path = f"{segmented_predictions_id}/audios/"
    dest_base_path = f"{segmented_predictions_id}_paper_tree_structure/"
    max_files_per_folder = 10000
    num_threads = 20

    if os.path.isfile(tsv_file):
        data = pd.read_csv(tsv_file, delimiter='\t')
    else:
        df = pd.concat([pd.read_csv(file_name, delimiter='\t') for file_name in glob.glob(os.path.join(segmented_predictions_id, '*.tsv'))])
        data = df.loc[:, ~df.columns.str.contains('^Unnamed')].reset_index(drop=True)
        data.to_csv(tsv_file, sep='\t')

    # Group data by year and month and count the rows
    grouped_data = data.groupby(['year', 'month']).size().reset_index(name='count')

    # Calculate the number of estimated folders for each year/month combination
    grouped_data['estimated_folders'] = grouped_data['count'].apply(lambda x: math.ceil(x / max_files_per_folder))

    # Print the results
    print("Estimated folders per year/month combination:")
    print(grouped_data[['year', 'month', 'estimated_folders']])

    # Create a dictionary to store the counters for each year/month combination
    folder_counters = collections.defaultdict(int)

    # Create a Lock object for thread-safe access to folder_counters
    lock = Lock()

    file_infos = []

    for index, file_id, prediction_type, year, month in data[['file_id', 'prediction_type', 'year', 'month']].itertuples():
        file_info = (index, file_id, prediction_type, year, month, folder_counters, src_base_path, dest_base_path, max_files_per_folder, lock)
        file_infos.append(file_info)

    # Use thread_map to handle file copying tasks concurrently and update the progress bar
    thread_map(copy_file, file_infos, max_workers=num_threads, desc="Copying files", unit="file")
