import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the data
df = pd.read_csv('predictions_Recording_2022-04-02_13-31-01_lensample0.5_stride0.1_thresh0.0.csv', sep='\t').head(300)
print(df)

# Set the figure size and style
sns.set(rc={'figure.figsize': (100, 10)})
sns.set_style("whitegrid")

palette = sns.color_palette("husl", n_colors=len(df['pred_label'].unique()))  # Set the desired color palette

# Create the scatter plot
ax = sns.scatterplot(data=df, x='offset', y='pred_conf_SM', hue='pred_label', s=25, edgecolor='black', alpha=0.6, palette=palette)

# Create a color mapping for 'pred_label'
color_mapping = {label: color for label, color in zip(df['pred_label'].unique(), palette)}

# Initialize the lists
max_conf_points = []
current_group = [0]

# Find the maximum confidence point for each group of temporally close points with the same label
time_threshold = 0.5  # Adjust this value based on the desired closeness
for i in range(1, len(df)):
    if (abs(df.loc[i, 'offset'] - df.loc[i - 1, 'offset']) <= time_threshold) and (df.loc[i, 'pred_label'] == df.loc[i - 1, 'pred_label']):
        current_group.append(i)
        line_color = color_mapping[df.loc[i, 'pred_label']]
        plt.plot(df.loc[[i - 1, i], 'offset'], df.loc[[i - 1, i], 'pred_conf_SM'], linewidth=1, color=line_color)
    else:
        max_conf_idx = df.loc[current_group, 'pred_conf_SM'].idxmax()
        max_conf_points.append(max_conf_idx)
        current_group = [i]

# Add the max confidence point of the last group
max_conf_idx = df.loc[current_group, 'pred_conf_SM'].idxmax()
max_conf_points.append(max_conf_idx)

# Plot the maximum confidence points
ax.scatter(df.loc[max_conf_points, 'offset'], df.loc[max_conf_points, 'pred_conf_SM'], marker='x', s=100, color='red')


# Set axis labels, title, and legend
ax.set_xlabel('Time (s)', fontsize=14, labelpad=20)
ax.set_ylabel('Pred conf', fontsize=14, labelpad=20)
ax.set_title('Predictions', fontsize=20, pad=20)
ax.legend(title='Pred Label', title_fontsize=12, loc='upper right')

# Customize axis ticks
ax.tick_params(axis='both', which='major', labelsize=12)

# Save the plot to a file
plt.savefig('predictions_Recording_2022-04-02_13-31-01_lensample0.5_stride0.1_thresh0.0.png', bbox_inches='tight', dpi=300)
