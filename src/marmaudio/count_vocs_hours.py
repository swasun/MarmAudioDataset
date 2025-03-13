# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import soundfile as sf
import glob
import os


def get_duration(file_name):
    sampling_rate = 96000
    info = sf.info(file_name)
    if info.samplerate != sampling_rate:
        print(f'Error sampling rate {info.samplerate} != {sampling_rate} of {file_name}')
    return info.duration

def count_raw_recordings_hours():
    root_dir = '../../../../Marmoset/uncut_raw'
    secs = 0.0
    sampling_rate = 96000
    for file_name in tqdm(glob.glob(os.path.join(root_dir, '*.wav'))):
        try:
            info = sf.info(file_name)
        except:
            print(f'Error with the file structure of {file_name}')
            continue
        secs += info.duration
    print(f'Raw recordings total nb hours: {secs // 3600}')

def count_segmented_vocs_hours():
    root_dir = 'marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13_paper_tree_structure'

    # Get a list of all wav file names
    file_names = glob.glob(os.path.join(root_dir, '*', '*.wav'))

    # Use thread_map to get the duration of each file
    durations = thread_map(get_duration, file_names, max_workers=10)  # Adjust max_workers as needed

    # Sum up all durations to get the total duration
    total_secs = sum(durations)

    print(f'Segmented vocs total nb hours: {total_secs // 3600}')

if __name__ == "__main__":
    #count_raw_recordings_hours() # 997h
    count_segmented_vocs_hours()
