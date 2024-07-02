# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import argparse
import soundfile as sf
from tqdm import tqdm
import glob
import os
from concurrent.futures import ThreadPoolExecutor


def decompress_flac_file(src_path, dest_path):
    data, samplerate = sf.read(src_path)
    sf.write(dest_path, data, samplerate)

def process_file(file_path, dest_folder):
    dir_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path).replace('.flac', '.wav')
    output_folder = os.path.join(dest_folder, dir_name)
    output_file_path = os.path.join(output_folder, file_name)
    if os.path.isfile(output_file_path):
        return
    os.makedirs(output_folder, exist_ok=True)
    decompress_flac_file(file_path, output_file_path)

def copy_tree_decompress_with_flac(src_folder, dest_folder):
    file_paths = glob.glob(os.path.join(src_folder, '**', '*.flac'), recursive=True)
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, file_paths, [dest_folder] * len(file_paths)), total=len(file_paths)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--output_folder_path', type=str, default='Decompressed_Vocalizations')
    args = parser.parse_args()

    copy_tree_decompress_with_flac(args.folder_path, args.output_folder_path)
