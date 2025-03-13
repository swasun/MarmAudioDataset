# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import pandas as pd
import shutil
import os
from tqdm import tqdm


if __name__ == "__main__":
    segmented_predictions_id = 'marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13'
    tsv_file = f"{segmented_predictions_id}_fixed2.tsv"
    audio_src_base_path = f"{segmented_predictions_id}/audios/"
    image_src_base_path = f"{segmented_predictions_id}/images/"
    samples_per_type = 100
    audio_dest_base_path = f"{segmented_predictions_id}_validation_data_23-05-20_n{samples_per_type}/audios"
    image_dest_base_path = f"{segmented_predictions_id}_validation_data_23-05-20_n{samples_per_type}/images"

    os.makedirs(audio_dest_base_path, exist_ok=True)
    os.makedirs(image_dest_base_path, exist_ok=True)

    # Read the TSV file using pandas
    data = pd.read_csv(tsv_file, delimiter='\t').sample(frac=1)

    # Group the data by prediction_type
    grouped_data = data.groupby(['prediction_type'])

    # Initialize a list to store the sampled files
    sampled_files = []

    """# Iterate through the grouped data and sample files per prediction type
    for prediction_type, group in tqdm(grouped_data):
        prediction_type = 'Infant_cry' if prediction_type == 'Infant cry' else prediction_type
        year_grouped_data = group.groupby(['year'])

        # Create a list to store the sampled indexes
        sampled_indexes = []

        # Calculate the number of samples per year
        years = len(year_grouped_data)
        samples_per_year = samples_per_type // years
        remaining_samples = samples_per_type % years

        # Sample files evenly across years
        for year, year_group in year_grouped_data:
            n_samples = samples_per_year
            if remaining_samples > 0:
                n_samples += 1
                remaining_samples -= 1

            if len(year_group) > 0:
                year_sampled_indexes = year_group.sample(n=min(n_samples, len(year_group)), replace=False).index
                sampled_indexes.extend(year_sampled_indexes)"""

    # Iterate through the grouped data and sample files per prediction type
    for prediction_type, group in tqdm(grouped_data):
        prediction_type = 'Infant_cry' if prediction_type == 'Infant cry' else prediction_type
        year_grouped_data = group.groupby(['year'])

        # Create a list to store the sampled indexes
        sampled_indexes = []

        # Calculate the number of samples per year
        years = len(year_grouped_data)
        samples_per_year = samples_per_type // years
        remaining_samples = samples_per_type % years

        # Sample files evenly across years
        for year, year_group in year_grouped_data:
            n_samples = samples_per_year
            if remaining_samples > 0:
                n_samples += 1
                remaining_samples -= 1

            if len(year_group) > 0:
                year_sampled_indexes = year_group.sample(n=min(n_samples, len(year_group)), replace=False).index
                sampled_indexes.extend(year_sampled_indexes)

        # If we haven't sampled enough files, randomly sample the remaining ones
        if len(sampled_indexes) < samples_per_type:
            additional_samples_needed = samples_per_type - len(sampled_indexes)
            additional_samples = group.drop(sampled_indexes).sample(n=min(additional_samples_needed, len(group)), replace=False).index
            sampled_indexes.extend(additional_samples)

        # Iterate through the sampled indexes and copy the files
        for index in tqdm(sampled_indexes):
            row = data.loc[index]
            file_id = row['file_id']
            year, month = row['year'], row['month']
            src_audio_path = os.path.join(audio_src_base_path, prediction_type, f"{file_id}.wav")
            src_image_path = os.path.join(image_src_base_path, prediction_type, f"{file_id}.png")
            dest_audio_path = os.path.join(audio_dest_base_path, f"{prediction_type}_{index}.wav")
            dest_image_path = os.path.join(image_dest_base_path, f"{prediction_type}_{index}.png")

            sampled_files.append((src_audio_path, dest_audio_path, src_image_path, dest_image_path))

    # Copy the sampled files to the destination folder
    for src_audio, dest_audio, src_image, dest_image in tqdm(sampled_files):
        shutil.copy2(src_audio, dest_audio)
        shutil.copy2(src_image, dest_image)
