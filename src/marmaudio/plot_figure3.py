import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os
from tqdm import tqdm


def plot_points(df):
    figscat = plt.figure(figsize=(20, 20))
    ax = figscat.add_subplot(1, 1, 1)

    ax.scatter(df.umap_x, df.umap_y, s=1, alpha=.025, c='grey')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    plt.savefig('projection_points_marmoset.pdf', bbox_inches='tight', dpi=300)

def plot_points_with_labels(df):
    figscat = plt.figure(figsize=(20, 20))
    ax = figscat.add_subplot(1, 1, 1)

    legend_elements = []

    for c, grp in df.groupby('label'):
        if c == 'Vocalization':
            ax.scatter(grp.umap_x, grp.umap_y, s=1, alpha=.025, c='grey', rasterized=False)
        else:
            scatter = ax.scatter(grp.umap_x, grp.umap_y, s=0.5, alpha=.1, c=None, rasterized=False)
            facecolor = scatter.get_facecolor()[0]
            facecolor_with_alpha = (facecolor[0], facecolor[1], facecolor[2], 0.5)
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=c,
                                        markerfacecolor=facecolor_with_alpha, markersize=8))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.legend(handles=legend_elements, frameon=False)

    plt.savefig('projection_points_with_labels_with_legend.jpg', bbox_inches='tight', dpi=300, transparent=True)

def plot_labels_legend(df):
    import matplotlib.font_manager as fm
    font_path = 'arial.ttf'
    font_prop = fm.FontProperties(fname=font_path, size=16)

    figscat = plt.figure(figsize=(20, 20))
    ax = figscat.add_subplot(1, 1, 1)

    legend_elements = []

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']  # get the default color cycle

    prediction_types = ['Infant cry', 'Phee', 'Seep', 'Trill', 'Tsik', 'Twitter']  # Desired order

    for idx, prediction_type in enumerate(prediction_types):
        if prediction_type in df['prediction_type'].unique():
            color = color_cycle[idx % len(color_cycle)]  # wrap around color cycle if necessary
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=prediction_type,
                                        markerfacecolor=color, markersize=8))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.legend(handles=legend_elements, frameon=False, prop=font_prop)

    plt.savefig('labels_legend.png', bbox_inches='tight', dpi=300)

def plot_imgs(df, segmented_predictions_id):
    df = df[df['prediction_type'] != 'Vocalization']

    # Standardize the data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(df[['umap_x', 'umap_y']])

    # Initialize the MiniBatchKMeans class
    mbk = MiniBatchKMeans(n_clusters=1000, random_state=42, batch_size=1000)

    # Fit it to your original (standardized) coordinates
    mbk.fit(X_std)

    # Get the coordinates of the cluster centers
    cluster_centers = mbk.cluster_centers_

    # Initialize the NearestNeighbors class
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)

    # Fit it to your original (standardized) coordinates
    nn.fit(X_std)

    # Find the nearest original point for each cluster center
    distances, indices = nn.kneighbors(cluster_centers)

    # 'indices' now contains the index of the nearest original point in 'df' for each cluster center
    df = df.iloc[indices.flatten()]

    fig, ax = plt.subplots(figsize=(25, 25))

    # Plot spectrograms instead of points
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        prediction_type = 'Infant_cry' if row['prediction_type'] == 'Infant cry' else row['prediction_type']
        x, y = row['umap_x'], row['umap_y']

        spectrogram_path = os.path.join(segmented_predictions_id, 'spectrograms_cleaned_new_cmap', f'{prediction_type}_{index}.png')
        if not os.path.exists(spectrogram_path):
            print(f"Spectrogram file does not exist: {spectrogram_path}")
            continue

        spectrogram = mpimg.imread(spectrogram_path)
        
        # Checking image transparency
        if spectrogram.shape[2] == 4 and np.all(spectrogram[:,:,3] == 0):
            print(f"Image {spectrogram_path} is fully transparent")

        im = OffsetImage(spectrogram, zoom=0.02)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    # This will help in scaling the axes according to the data points
    ax.autoscale_view()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    plt.savefig('projection_spectrograms_cleaned_1000_bigger.png', bbox_inches='tight', dpi=300)

def plot_imgs_spreaded(df, segmented_predictions_id):
    import networkx as nx

    df = df[df['prediction_type'] != 'Vocalization']

    # Standardize the data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(df[['umap_x', 'umap_y']])

    # Initialize the MiniBatchKMeans class
    mbk = MiniBatchKMeans(n_clusters=500, random_state=42, batch_size=1000)

    # Fit it to your original (standardized) coordinates
    mbk.fit(X_std)

    # Get the coordinates of the cluster centers
    cluster_centers = mbk.cluster_centers_

    # Initialize the NearestNeighbors class
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)

    # Fit it to your original (standardized) coordinates
    nn.fit(X_std)

    # Find the nearest original point for each cluster center
    distances, indices = nn.kneighbors(cluster_centers)

    # 'indices' now contains the index of the nearest original point in 'df' for each cluster center
    df = df.iloc[indices.flatten()]

    # Save the original indices in a new column
    df['original_index'] = df.index

    # Reset the index of the DataFrame to align it with the indices of the graph nodes
    df = df.reset_index(drop=True)

    # create a graph connecting each data point to its nearest neighbors
    G = nx.Graph()

    for i, (x, y) in enumerate(X_std[indices.flatten()]):
        # add a node for this data point
        G.add_node(i, pos=(x, y))

    # compute the spring layout
    pos = nx.spring_layout(G)

    fig, ax = plt.subplots(figsize=(20, 20))

    # Plot spectrograms instead of points
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        prediction_type = 'Infant_cry' if row['prediction_type'] == 'Infant cry' else row['prediction_type']
        x, y = pos[index]  # get the new position from the force-directed layout

        spectrogram_path = os.path.join(segmented_predictions_id, 'spectrograms_cleaned_new_cmap', f'{prediction_type}_{row["original_index"]}.png')
        if not os.path.exists(spectrogram_path):
            print(f"Spectrogram file does not exist: {spectrogram_path}")
            continue

        spectrogram = mpimg.imread(spectrogram_path)
        
        # Checking image transparency
        if spectrogram.shape[2] == 4 and np.all(spectrogram[:,:,3] == 0):
            print(f"Image {spectrogram_path} is fully transparent")

        im = OffsetImage(spectrogram, zoom=0.02)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    # This will help in scaling the axes according to the data points
    ax.autoscale_view()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    plt.savefig('projection_spectrograms_cleaned_500_spreaded.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--encodings_file_path', nargs='?', type=str, required=True)
    parser.add_argument('--dataset_root_path', nargs='?', type=str, required=True)
    args = parser.parse_args()

    encodings = np.load(args.encodings_file_path, allow_pickle=True).item()
    idxs, umap = encodings['idx'], encodings['umap']
    tsv_file = f"{args.dataset_root_path}/Annotations.tsv"
    df = pd.read_csv(tsv_file, sep='\t')
    idxs = [os.path.basename(idx) for idx in idxs]

    df = df[df['file_name'].isin(idxs)].reset_index(drop=True)
    df[['umap_x', 'umap_y']] = umap

    #plot_points(df)
    plot_points_with_labels(df)
    #plot_imgs(df, segmented_predictions_id)
    #plot_labels_legend(df)
