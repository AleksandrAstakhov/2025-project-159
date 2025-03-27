import h5py
import os
import torch
import scipy.io as sio
import re

import numpy as np

import torch.nn as nn
import torch.optim as optim

from classifier import EmoClassifier

from torch.utils.data import Dataset

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.utils import to_dense_adj

from tqdm.auto import tqdm as tqdma


class SEEDIVDataset(Dataset):
    def __init__(
        self,
        h5_path,
        raw_data_dir,
        time_step_size,
        stride,
        max_seq_len,
        threshold,
        graphs_with_loops,
        can_load=False,
    ):

        labels = [
            [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
        ]

        self.h5_path = h5_path
        self.time_step_size = time_step_size
        self.stride = stride
        self.max_seq_len = max_seq_len
        self.graph_with_loops = graphs_with_loops
        self.dataset_size = 0
        self.threshold = threshold

        if can_load:
            with h5py.File(h5_path, "r") as f:
                self.dataset_size = f["dataset"]["dataset_size"][()]
                return

        with h5py.File(h5_path, "w") as f:
            eeg_group = f.require_group("dataset")

        for session in tqdma(range(1, 4), desc="Processing sessions"):
            data_files = []

            session_data = os.path.join(raw_data_dir, str(session))

            for f in os.listdir(session_data):
                if f.endswith(".mat"):
                    data_files.append(os.path.join(session_data, f))

            for file in tqdma(data_files, desc="Processing files"):
                eeg_raw_data = sio.loadmat(file)

                for trial in tqdma(range(1, 25), desc="Processing trials", leave=False):
                    base = re.sub(r"\d", "", list(eeg_raw_data.keys())[3])

                    eeg_signal = torch.tensor(eeg_raw_data[f"{base}{trial}"])

                    eeg_windows = eeg_signal.unfold(
                        dimension=1, size=time_step_size, step=stride
                    ).permute(1, 0, 2)

                    begin_seq = 0
                    end_seq = max_seq_len

                    while end_seq <= len(eeg_windows):
                        self._process_windows_seq(
                            eeg_windows[begin_seq:end_seq],
                            labels[session - 1][trial - 1],
                        )

                        begin_seq += max_seq_len
                        end_seq += max_seq_len

        with h5py.File(h5_path, "a") as f:
            f["dataset"].create_dataset("dataset_size", data=self.dataset_size)

    def _get_adj_mat(self, window):
        signals_centered = window - window.mean(dim=1, keepdim=True)

        cov_matrix = torch.matmul(signals_centered, signals_centered.T) / (
            self.time_step_size - 1
        )

        std_devs = torch.std(window, dim=1, keepdim=True)

        epsilon = 1e-8
        correlation_matrix = cov_matrix / ((std_devs * std_devs.T) + epsilon)

        correlation_matrix = torch.where(
            torch.abs(correlation_matrix) < self.threshold,
            torch.tensor(0.0, device=correlation_matrix.device),
            correlation_matrix,
        )

        return (
            correlation_matrix
            if self.graph_with_loops
            else correlation_matrix - torch.eye(correlation_matrix.shape[0])
        )

    def _process_windows_seq(self, windows_seq, label):
        with h5py.File(self.h5_path, "a") as f:
            dataset = f["dataset"]

            sample = dataset.require_group(str(self.dataset_size))
            sample.create_dataset("label", data=label)

            self.dataset_size += 1

            for idx, window in enumerate(windows_seq):
                adj_mat = self._get_adj_mat(window)

                edge_idx, edge_attr = dense_to_sparse(adj_mat)

                sample.create_dataset(f"x{idx}", data=window.numpy())
                sample.create_dataset(f"edge_idx{idx}", data=edge_idx.numpy())
                sample.create_dataset(f"edge_attr{idx}", data=edge_attr.numpy())

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            sample = f["dataset"][f"{idx}"]

            return [
                (
                    Data(
                        x=torch.tensor(np.array(sample[f"x{id}"])),
                        edge_index=torch.tensor(np.array(sample[f"edge_idx{id}"])),
                        edge_attr=torch.tensor(np.array(sample[f"edge_attr{id}"])),
                    ),
                    torch.tensor(sample["label"][()]),
                )
                for id in range(self.max_seq_len)
            ]
