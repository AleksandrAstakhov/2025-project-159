import os
import re
import torch
import zarr
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from data_api import DataAPI


raw_data_dir = "/home/alex/workspace/project-159/SEED_IV/archive/eeg_raw_data"
chunk_size = 20
time_step_size = 10
step = 5

labels = np.array(
    [
        [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
    ]
)

dataset = DataAPI("coo_optim.zarr", group_shape=(chunk_size, 62, time_step_size))


def get_adj_matrices(windows, threshold=0.7):
    balanced = (windows - windows.mean(dim=2, keepdim=True)) / (
        windows.std(dim=2, keepdim=True) + 1e-6
    )
    adj = torch.einsum("ink,imk->inm", balanced, balanced) / (windows.shape[2] - 1)

    # Убираем единички на диагонали
    adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))

    # Применяем пороговую обработку
    adj = torch.where(adj > threshold, adj, torch.zeros_like(adj))

    return adj


for session in tqdm(range(1, 2), desc="Processing sessions"):
    session_data = os.path.join(raw_data_dir, str(session))
    data_files = [
        os.path.join(session_data, f)
        for f in os.listdir(session_data)
        if f.endswith(".mat")
    ]

    for file in tqdm(data_files, desc="Processing files"):
        eeg_raw_data = sio.loadmat(file)
        base = re.sub(r"\d", "", list(eeg_raw_data.keys())[3])

        for trial in tqdm(range(1, 25), desc="Processing trials", leave=False):
            eeg_signal = torch.tensor(eeg_raw_data.get(f"{base}{trial}", np.array([])))
            if eeg_signal.numel() == 0:
                continue

            windows = eeg_signal.unfold(1, time_step_size, step).permute(1, 0, 2)[:-1]
            adj_matrices = get_adj_matrices(windows)
            sparse_adj_matrices = [sp.coo_matrix(matrix) for matrix in adj_matrices]
            windows = windows.numpy()

            label = labels[session - 1, trial - 1]
            for i in range(0, len(windows) - chunk_size + 1, chunk_size):
                dataset.add_group(
                    sparse_matrices=sparse_adj_matrices[i : i + chunk_size],
                    dense_matrices=windows[i : i + chunk_size],
                    label=label,
                )
