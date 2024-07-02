import matplotlib
import utils as u
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import models
import os
import numpy as np
import pandas as pd
import hdbscan
import argparse
import shutil
import pathlib
try:
    import sounddevice as sd
    soundAvailable = True
except:
    soundAvailable = False


"""
python sort_cluster.py encodings_AE_marmosset_logMel128_16feat_all-vocs.npy W:\banco\data\sounds\Marmoset vocalizations --dataloader_n_jobs=1 --root_experiment_folder=. --detection_pkl_file_name=detections_1910_fixed_positions.pkl

Fix for hdbscan error on Windows: pip install joblib==1.1.0
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, \
        description="""Interface to visualize projected vocalizations (UMAP reduced AE embeddings), tune HDBSCAN parameters, and browse clusters by clicking on projected points.\n
        If satisfying parameters are reached, the clusters can be plotted in .png folders by typing y after closing the projection plot.\n
        For insights on how to tune HDBSCAN parameters, read https://hdbscan.readthedocs.io/en/latest/parameter_selection.html.\n
        To enable sound playing when browsing points, make sure the sounddevice package is installed.""")
    parser.add_argument('encodings', type=str, help='.npy file containing umap projections and their associated index in the detection.pkl table (built using compute_embeddings.py)')
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--min_cluster_size', type=int, default=400, help='Used for HDBSCAN clustering.')
    parser.add_argument('--hdbscan_n_jobs', type=int, default=-1)
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    parser.add_argument('--min_sample', type=int, default=20, help='Used for HDBSCAN clustering.')
    parser.add_argument('--eps', type=float, default=0.05, help='Used for HDBSCAN clustering.')
    parser.add_argument('--padding_duration', type=float, default=0.5)
    parser.add_argument('--sampling_rate', type=int, default=96000)
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, default='detections_1910_fixed_positions.pkl')
    parser.add_argument('--species_name', type=str, default='marmoset')
    args = parser.parse_args()

    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frontend = models.get['frontend'](args.sampling_rate, 1024, args.padding_duration, 128).to(gpu)

    encodings = np.load(os.path.join(args.root_experiment_folder, args.encodings), allow_pickle=True).item()
    idxs, umap = encodings['idx'], encodings['umap']
    df = pd.read_pickle(os.path.join(args.root_experiment_folder, args.detection_pkl_file_name))
    df = df.loc[idxs]
    # Use HDBSCAN to cluster the embedings (min_cluster_size and min_samples parameters need to be tuned)
    df.loc[idxs, 'cluster'] = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                            min_samples=args.min_sample,
                                            core_dist_n_jobs=args.hdbscan_n_jobs,
                                            cluster_selection_epsilon=args.eps,
                                            cluster_selection_method='leaf').fit_predict(umap)
    df.loc[idxs, ['umap_x', 'umap_y']] = umap
    df.cluster = df.cluster.astype(int)

    figscat = plt.figure(figsize=(20, 10))
    plt.title(f'{args.encodings} {args.min_cluster_size} {args.min_sample} {args.eps}')
    for c, grp in df.groupby('cluster'):
        plt.scatter(grp.umap_x, grp.umap_y, s=3, alpha=.1, c='grey' if c == -1 else None)

    plt.tight_layout()
    axScat = figscat.axes[0]
    plt.savefig(os.path.join(args.root_experiment_folder, 'projection.png'))
    figSpec = plt.figure()
    plt.scatter(0, 0)
    axSpec = figSpec.axes[0]
    plt.close()

    print(df.groupby('cluster').count().onset)

    """class temp():
        def __init__(self):
            self.row = ""
        def onclick(self, event):
            # find the closest point to the mouse
            left, right, bottom, top = axScat.get_xlim()[0], axScat.get_xlim()[1], axScat.get_ylim()[0], axScat.get_ylim()[1]
            rangex, rangey =  right - left, top - bottom
            closest = (np.sqrt(((df.umap_x - event.xdata)/rangex)**2 + ((df.umap_y  - event.ydata)/rangey)**2)).idxmin()
            row = df.loc[closest]
            # read waveform and compute spectrogram
            info = sf.info(f'{args.audio_folder}/{row.parent_name}.wav')
            #dur, fs = info.duration, info.samplerate
            #start = int(np.clip(row.pos - args.padding_duration/2, 0, dur - args.padding_duration) * fs)
            #print(row)
            #sig, fs = sf.read(f'{args.audio_folder}/{row.parent_name}.wav', start=start, stop=start + int(args.padding_duration*fs), always_2d=True)
            #sig = sig[:,0]
            padding_samples = (args.padding_duration * args.sampling_rate) // 2
            sig, fs = sf.read(f'{args.audio_folder}/{row.parent_name}.wav', start=int(row.pos - padding_samples), stop=int(row.pos + padding_samples))
            sig = sig[:,0]
            spec = frontend(torch.Tensor(sig).to(gpu).view(1, -1).float()).detach().cpu().squeeze()
            axSpec.clear()
            axSpec.imshow(spec, origin='lower', aspect='auto')
            # Display and metadata management
            axSpec.set_title(f'{closest}, cluster {row.cluster} ({(df.cluster==row.cluster).sum()} points)')
            axScat.scatter(row.umap_x, row.umap_y, c='r')
            axScat.set_xlim(left, right)
            axScat.set_ylim(bottom, top)
            figSpec.canvas.draw()
            figscat.canvas.draw()
            # Play the audio
            if soundAvailable:
                sd.play(sig*10, fs)

    mtemp = temp()
    cid = figscat.canvas.mpl_connect('button_press_event', mtemp.onclick)
    plt.show()

    if input('\nType y to print cluster pngs.\n/!\ the cluster_pngs folder will be reset, backup if needed /!\ ') != 'y':
        exit()"""

    if os.path.isdir('cluster_pngs'):
        shutil.rmtree(os.path.join(args.root_experiment_folder, 'cluster_pngs/*'), ignore_errors=True)
    else:
        pathlib.Path(os.path.join(args.root_experiment_folder, 'cluster_pngs')).mkdir(parents=True, exist_ok=True)

    for c, grp in df.groupby('cluster'):
        """if c == -1 or len(grp) > 10_000:
            continue"""
        pathlib.Path(os.path.join(args.root_experiment_folder, 'cluster_pngs', str(c))).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for x, idx in tqdm(torch.utils.data.DataLoader(u.Dataset(grp.sample(200), args.audio_folder, sampleDur=.5, species_name=args.species_name), batch_size=1, num_workers=args.dataloader_n_jobs), leave=False, desc=str(c)):
                x = x.to(gpu)
                x = frontend(x).cpu().detach().squeeze()
                plt.imshow(x, origin='lower', aspect='auto')
                plt.savefig(f'{args.root_experiment_folder}/cluster_pngs/{c}/{idx.squeeze().item()}')
                plt.close()
