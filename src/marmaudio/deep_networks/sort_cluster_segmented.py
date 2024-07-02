import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import hdbscan
import shutil
import os
import pathlib
from tqdm import tqdm


if __name__ == "__main__":
    encodings = np.load('encodings_segmented_AE_logMel128_16feat.npy', allow_pickle=True).item()
    idxs, umap = encodings['idx'], encodings['umap']
    df = pd.read_csv('voc_vs_noise_segmented_filtered_predictions_dynamic_2023-05-13_all.tsv', sep='\t')
    root_experiment_folder = 'MarmosetVocalizations'

    min_cluster_size = 400
    min_sample = 20
    hdbscan_n_jobs = -1
    eps = 0.05

    df = df.loc[idxs]
    df.loc[idxs, ['umap_x', 'umap_y']] = umap

    df.loc[idxs, 'cluster'] = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                            min_samples=min_sample,
                                            core_dist_n_jobs=hdbscan_n_jobs,
                                            cluster_selection_epsilon=eps,
                                            cluster_selection_method='leaf').fit_predict(umap)
    df.cluster = df.cluster.astype(int)

    # Choose a colormap
    cmap = cm.get_cmap('tab20c')

    # Normalize cluster numbers to [0,1] to map to the colormap
    normalize = plt.Normalize(vmin=df['cluster'].min(), vmax=df['cluster'].max())

    labels = []
    lines = []

    figscat = plt.figure(figsize=(20, 20))
    ax = figscat.add_subplot(1, 1, 1)
    ax.set_title(f'{min_cluster_size} {min_sample} {eps}')
    for c, grp in df.groupby('cluster'):
        if c != -1:  # Assuming -1 is noise
            color = cmap(normalize(c))
        else:
            color = 'grey'
        scatter = ax.scatter(grp.umap_x, grp.umap_y, s=1, alpha=.025, c=[color])
        if c != -1:
            labels.append(f'Cluster {c}')
            lines.append(mlines.Line2D([], [], color=color, markersize=15, label=f'Cluster {c}'))

    ax.legend(handles=lines, loc='best')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    plt.savefig('projection_points_clusters.png', bbox_inches='tight', dpi=300)
    plt.close()

    if os.path.isdir('cluster_pngs'):
        shutil.rmtree(os.path.join(root_experiment_folder, 'cluster_pngs/*'), ignore_errors=True)
    else:
        pathlib.Path(os.path.join(root_experiment_folder, 'cluster_pngs')).mkdir(parents=True, exist_ok=True)

    for c, grp in df.groupby('cluster'):
        """if c == -1 or len(grp) > 10_000:
            continue"""
        pathlib.Path(os.path.join(root_experiment_folder, 'cluster_pngs', str(c))).mkdir(parents=True, exist_ok=True)
        df2 = grp.sample(15)
        for index, row in tqdm(df2.iterrows(), total=df2.shape[0]):
            shutil.copyfile(os.path.join('voc_vs_noise_segmented_filtered_predictions_dynamic_2023-05-13/images/Vocalization', row['file_id'] + '.png'),
                os.path.join(root_experiment_folder, 'cluster_pngs', str(c), row['file_id'] + '.png'))
