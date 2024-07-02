# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paul_DOT_best_AT_lis-lab_DOT_fr>

# License: BSD (3-clause)

import torch
from models import get
import numpy as np
import umap
from tqdm import tqdm
import argparse
import os
import soundfile as sf
import glob


class Dataset(torch.utils.data.Dataset):

    def __init__(self, src_folder, sample_duration, sampling_rate):
        super(Dataset, self)
        self.file_paths = glob.glob(os.path.join(src_folder, '*', '*.wav'))
        self.indices = np.arange(len(self.file_paths))
        self.norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)
        self.sample_duration = sample_duration
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sig, _ = sf.read(self.file_paths[idx], always_2d=True)

        middle_point = len(sig) // 2
        ms_samples = int(self.sample_duration * self.sampling_rate)
        start_index = middle_point - ms_samples // 2
        end_index = middle_point + ms_samples // 2

        sig = sig[start_index:end_index]

        return torch.tensor(self.norm(sig)).float().squeeze(), self.indices[idx]


"""
python compute_embeddings_segmented.py AE_marmosset_logMel128_16feat_all-vocs_160323.stdc marmoset_segmented_filtered_predictions_dynamic_and_manual_offsets_2023-05-13_paper_tree_structure
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute the AE projection of vocalizations once it was trained.")
    parser.add_argument('modelname', type=str, help='Filename of the AE weights (.stdc)')
    parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
    parser.add_argument('--sampling_rate_khz', type=str, default='96')
    parser.add_argument('--dataloader_n_jobs', type=int, default=8)
    parser.add_argument('--sample_duration', type=float, default=.5)
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.sampling_rate_khz == '96':
        sampling_rate = 96_000
    elif args.sampling_rate_khz == '44_1':
        sampling_rate = 44_100

    nfeat = 256
    frontend = get[f'frontend'](sampling_rate, 1024, args.sample_duration, 128).eval()
    encoder = get['sparrow_encoder'](nfeat // 16, (4, 4))
    decoder = get['sparrow_decoder'](nfeat, (4, 4))
    model = torch.nn.Sequential(frontend, encoder, decoder).to(device)
    model.load_state_dict(torch.load(os.path.join(args.modelname), map_location=device))

    print('Computing AE projections...')
    loader = torch.utils.data.DataLoader(Dataset(args.audio_folder, args.sample_duration, sampling_rate), \
        batch_size=64, shuffle=False, num_workers=args.dataloader_n_jobs, prefetch_factor=8)
    with torch.no_grad():
        encodings, idxs = [], []
        for x, idx in tqdm(loader):
            encoding = model[:2](x.to(device))
            idxs.extend(idx)
            encodings.extend(encoding.cpu().detach())
    idxs = np.array(idxs)
    encodings = np.stack(encodings)

    print('Computing UMAP projections...')
    X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
    np.save(f'encodings_segmented_'+os.path.basename(args.modelname)[:-4]+'npy', {'encodings':encodings, 'idx':idxs, 'umap':X})
