import pandas as pd
import os
import numpy as np
import pathlib
from tqdm import tqdm
import shutil
import tarfile


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


if __name__ == "__main__":
    percentage = 1
    source_root_path = os.path.join('results', 'MarmosetVocalizations_all')
    destination_root_path = source_root_path + f'_21-01-22_p{percentage}'

    df = pd.read_csv(os.path.join(source_root_path, 'labels.tsv'), sep='\t', low_memory=False, index_col=0).reset_index(drop=True)
    df = df.drop(df.query('parent_name == sample_name').index).reset_index(drop=True)

    labels_root_path = 'results/labels/MarmosetVocalizations'
    tolabels_root_path = 'results/tolabels/MarmosetVocalizations'

    sound_ids_to_filter = []
    for file_name in os.listdir(labels_root_path):
        current_df = pd.read_csv(os.path.join(labels_root_path, file_name), sep='\t')
        sound_ids_to_filter += current_df['id'].tolist()

    if len(sound_ids_to_filter) > 0:
        print(f'Filtering {len(sound_ids_to_filter)} sound ids from the labels...')
        indices_to_filter = list()
        for sound_id in tqdm(sound_ids_to_filter, desc='Filtering already labelled'):
            sample_name, relative_id = sound_id.split('_sample_')
            found_row = df[(df.sample_name == sample_name) & (df.relative_id == int(relative_id))]
            assert(len(found_row) <= 1)
            if len(found_row) > 0:
                indices_to_filter.append(found_row.index[0])
        print(f'Before filtering already labelled: #{len(df)}')
        df = df.drop(indices_to_filter).reset_index(drop=True)
        print(f'After filtering already labelled: #{len(df)}')

    for file_name in os.listdir(tolabels_root_path):
        current_df = pd.read_csv(os.path.join(tolabels_root_path, file_name), sep='\t')
        
        print(f'Filtering {len(current_df)} samples to be labelled...')
        print(f'Before filtering to be labelled: #{len(df)}')
        df['key1'] = 1
        current_df['key2'] = 1
        df = pd.merge(df, current_df, on=['relative_id', 'parent_name', 'sample_name'], how = 'left')
        df = df[~(df.key2 == df.key1)]
        df = df.drop(['key1','key2'], axis=1)
        current_df = current_df.drop(['key2'], axis=1)
        df.drop(df.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')].reset_index(drop=True)
        df.columns = df.columns.str.rstrip("_x")
        print(f'After filtering to be labelled: #{len(df)}')

    indices_number = int(percentage / 100 * len(df))
    indices = np.arange(indices_number)
    print(f'Picking {len(indices)} samples from {len(df)} available...')
    np.random.shuffle(indices)

    df = df.loc[indices]

    pathlib.Path(os.path.join(destination_root_path, 'audios')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(destination_root_path, 'images')).mkdir(parents=True, exist_ok=True)
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Copying'):
        source_audio_file_path = os.path.join(source_root_path, 'audios', f"{row['sample_name']}_sample_{row['relative_id']}.wav")
        source_image_file_path = os.path.join(source_root_path, 'images', f"{row['sample_name']}_sample_{row['relative_id']}.png")
        destination_audio_file_path = os.path.join(destination_root_path, 'audios', f"{row['sample_name']}_sample_{row['relative_id']}.wav")
        destination_image_file_path = os.path.join(destination_root_path, 'images', f"{row['sample_name']}_sample_{row['relative_id']}.png")
        shutil.copyfile(source_audio_file_path, destination_audio_file_path)
        shutil.copyfile(source_image_file_path, destination_image_file_path)

    labels_file_name = f'{os.path.basename(destination_root_path)}_labels.tsv'
    labels_file_path = os.path.join(destination_root_path, labels_file_name)
    df.to_csv(labels_file_path, sep='\t')
    shutil.copyfile(labels_file_path, os.path.join(tolabels_root_path, labels_file_name))

    print('Creating tarfile...')
    make_tarfile(output_filename=destination_root_path + '.tar.gz', source_dir=destination_root_path)
