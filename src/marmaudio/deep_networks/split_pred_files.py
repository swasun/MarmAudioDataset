import os
import pandas as pd

input_tsv_path = "predictions_Recording_2023_all_lensample0.5_stride0.1_thresh0.0.csv"

# Read the input TSV file
data = pd.read_csv(input_tsv_path, sep='\t')

# Group the data by the 'fn' column
grouped_data = data.groupby('fn')

output_folder = "predictions_lensample0.5_stride0.1_thresh0.0"

# Iterate over the groups and save each group as a separate TSV file
for fn, group in grouped_data:
    # Remove the '.wav' extension from the file name
    base_fn = os.path.splitext(fn)[0]
    
    # Create the output file name based on the pattern
    output_file_name = f"predictions_{base_fn}_lensample0.5_stride0.1_thresh0.0.csv"
    
    # Create the full output file path
    output_file_path = os.path.join(output_folder, output_file_name)
    
    # Save the group to a TSV file
    group.to_csv(output_file_path, index=False, sep='\t')
