# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = pd.read_csv('marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_07-05-23_all.tsv', sep='\t')

    data = data[data.prediction_type != 'Vocalization']

    print(data)
    print(data.prediction_type.value_counts())

    # Convert year, month, and day columns into a single date column
    data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))

    # Group by date and prediction type, and count the number of occurrences
    grouped_data = data.groupby(['date', 'prediction_type']).size().reset_index(name='count')

    # Pivot the data to have prediction types as columns
    pivoted_data = grouped_data.pivot_table(index='date', columns='prediction_type', values='count', fill_value=0)

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 8))
    pivoted_data.plot(kind='bar', stacked=True, ax=ax)

    ax.set_ylabel("Number of Vocalizations")
    ax.set_xlabel("Date")
    # Rotate x-axis labels and adjust their alignment
    plt.xticks(rotation=45, ha='right')
    ax.set_title("Number of Each Type of Vocalization by Date")

    plt.savefig('labels_over_time_wo_vocalization.png', bbox_inches='tight', dpi=100)
