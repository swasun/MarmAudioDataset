# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import torch
import utils as u
from models import get
import numpy as np
import umap
from tqdm import tqdm
import pandas as pd
import argparse
import os


"""
python compute_embeddings.py AE_marmosset_logMel128_16feat_all-vocs.stdc /media/charly/9A0CEE3F0CEE1653/Charly/Vocalizations
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute the AE projection of vocalizations once it was trained.")
    parser.add_argument('modelname', type=str, help='Filename of the AE weights (.stdc)')
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--root_experiment_folder', type=str, default='.')
    parser.add_argument('--detection_pkl_file_name', type=str, default='detections_1910_fixed_positions.pkl')
    parser.add_argument('--species_name', type=str, default='marmoset')
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--sample_duration', type=float, default=.5)
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.sampling_rate_khz == '96':
        sampling_rate = 96_000
    elif args.sampling_rate_khz == '44_1':
        sampling_rate = 44_100

    #nfeat = 256
    nfeat = 16
    frontend = get[f'frontend'](sampling_rate, 1024, args.sample_duration, nfeat).eval()
    """encoder = get['sparrow_encoder'](nfeat // 16, (4, 4))
    decoder = get['sparrow_decoder'](nfeat, (4, 4))"""
    encoder = get['sparrow_encoder_old'](nfeat)
    decoder = get['sparrow_decoder_old'](nfeat, (4, 2))
    model = torch.nn.Sequential(frontend, encoder, decoder).to(device)
    model.load_state_dict(torch.load(os.path.join(args.modelname), map_location=device))

    #df = pd.read_pickle(os.path.join(args.root_experiment_folder, args.detection_pkl_file_name))
    df = pd.read_csv('/media/charly/9A0CEE3F0CEE1653/Charly/Annotations.tsv', sep='\t')

    print('Computing AE projections...')
    """loader = torch.utils.data.DataLoader(u.Dataset(df, args.audio_folder, sampleDur=args.sample_duration, species_name=args.species_name), \
                                         batch_size=64, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=8)"""
    marmoset_labels = df.label.unique()
    typetoidx_marmoset = {k:i for i, k in enumerate(marmoset_labels)}
    loader = torch.utils.data.DataLoader(u.FinalDataset(df, args.audio_folder, typetoidx_marmoset=typetoidx_marmoset, brown=False, sampleDur=.5, retType=True, species_name='marmoset'), batch_size=256, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=8)
    i = 0
    with torch.no_grad():
        encodings, idxs = [], []
        for x, idx, label in tqdm(loader):
            encoding = model[:2](x.to(device))
            idxs.extend(idx)
            encodings.extend(encoding.cpu().detach())
            """if i == 50:
                break
            i += 1"""
    idxs = np.array(idxs)
    encodings = np.stack(encodings)

    print('Computing UMAP projections...')
    X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
    np.save(f'{args.root_experiment_folder}/encodings_'+args.modelname[:-4]+'_10-05-25.npy', {'encodings':encodings, 'idx':idxs, 'umap':X})
