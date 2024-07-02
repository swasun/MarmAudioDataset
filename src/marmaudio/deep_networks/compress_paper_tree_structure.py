# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import os
import subprocess
from tqdm import tqdm


def compress_with_flac(src_path, dest_path):
    subprocess.run(['flac', src_path, '-o', dest_path, '--totally-silent', '--compression-level-8'])

def copy_tree_compress_with_flac(src_folder, dest_folder):
    for root, dirs, files in tqdm(os.walk(src_folder)):
        for file in tqdm(files):
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_folder, os.path.relpath(src_file, src_folder))
            dest_file = os.path.splitext(dest_file)[0] + '.flac'

            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            compress_with_flac(src_file, dest_file)

# Source folder path
src_folder = "marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13_paper_tree_structure"

# Destination folder path
dest_folder = f'{src_folder}_flac'

# Copy the file tree while compressing with FLAC
copy_tree_compress_with_flac(src_folder, dest_folder)
