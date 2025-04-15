import os
import re
import torch
import zarr
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from scipy.signal import hilbert, butter, filtfilt

from data_api import DataAPI

freq_bands = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (14, 30),
    "gamma": (31, 50),
}

fs = 200

raw_data_dir = "/home/alex/workspace/project-159/SEED_IV/archive/eeg_raw_data"
chunk_size = 6
time_step_size = 6 * fs
step = 200

labels = np.array(
    [
        [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
        [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
        [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
    ]
)

dataset = DataAPI("coo_optim_2_valid.zarr", group_shape=(chunk_size, 62, 5))


def get_adj_matrices(windows, threshold=0.7):
    balanced = (windows - windows.mean(dim=2, keepdim=True)) / (
        windows.std(dim=2, keepdim=True) + 1e-6
    )
    adj = torch.einsum("ink,imk->inm", balanced, balanced) / (windows.shape[2] - 1)

    adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))

    adj = torch.where(adj > threshold, adj, torch.zeros_like(adj))

    return adj


def compute_plv_matrix(signal, threshold=0.7):
    phase = np.angle(hilbert(signal, axis=-1))
    n_windows, n_channels, n_samples = phase.shape

    exp_phase = np.exp(1j * phase)
    plv_matrix = (
        np.abs(np.einsum("...ix,...jx->...ij", exp_phase, exp_phase.conj())) / n_samples
    )

    diag_mask = np.eye(n_channels, dtype=bool)
    plv_matrix[..., diag_mask] = 0

    if threshold is not None:
        plv_matrix[plv_matrix < threshold] = 0

    return plv_matrix


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def compute_de(X, fs, freq_bands):
    num_windows, num_channels, time_len = X.shape
    num_bands = len(freq_bands)
    de_features = np.zeros((num_windows, num_channels, num_bands))

    for band_idx, (band_name, (low, high)) in enumerate(freq_bands.items()):
        b, a = butter_bandpass(low, high, fs, order=4)
        filtered = filtfilt(b, a, X, axis=-1)

        sigma2 = np.var(filtered, axis=-1, ddof=1)
        de = 0.5 * np.log(2 * np.pi * np.e * sigma2)

        de_features[:, :, band_idx] = de

    return de_features


for session in tqdm(range(3, 4), desc="Processing sessions"):
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
            windows = windows.numpy()

            adj_matrices = compute_plv_matrix(windows)
            sparse_adj_matrices = [sp.coo_matrix(matrix) for matrix in adj_matrices]

            dense_data = compute_de(windows, fs=fs, freq_bands=freq_bands)

            label = labels[session - 1, trial - 1]
            for i in range(0, len(windows) - chunk_size + 1, chunk_size):
                dataset.add_group(
                    sparse_matrices=sparse_adj_matrices[i : i + chunk_size],
                    dense_matrices=dense_data[i : i + chunk_size],
                    label=label,
                )

dataset.flush()
