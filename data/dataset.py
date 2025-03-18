from torch.utils.data import Dataset

import scipy.io as sio
import torch
import os


class SEEDIVDataset(Dataset):
    def __init__(
        self,
        data_path,
        window_size,
        window_stride,
        windows_number,
        threshold,
        *,
        need_save=False,
        can_load=False,
        path_to_save=None,
    ):
        super().__init__()

        labels = [
            [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
        ]

        self.data_path = data_path
        self.threshold = threshold
        self.window_size = window_size
        self.window_stride = window_stride
        self.windows_number = windows_number

        self.eeg_samples = []

        if can_load and path_to_save is not None:
            self.eeg_samples = torch.load(path_to_save)

        for sessionn in range(1, 4):

            data_files = []

            folder_path = os.path.join(self.data_path, str(sessionn))
            for file in os.listdir(folder_path):
                if file.endswith(".mat"):
                    data_files.append(os.path.join(self.data_path, file))

            for mat_file in data_files:
                mat_dict = sio.loadmat(mat_file)

                for trial in range(1, 25):
                    eeg_signal = torch.tensor(mat_dict[f"cz_eeg{trial}"])

                    eeg_windows = eeg_signal.unfold(
                        dimension=1, size=3, step=3
                    ).permute(1, 0, 2)

                    for i in range(len(eeg_windows) // self.windows_number):
                        start_idx = i * self.windows_number
                        end_idx = (i + 1) * self.window_size
                        sample = eeg_windows[start_idx:end_idx]

                        adj_mat_sample = torch.tensor(
                            [
                                self._build_adjacency_matrix(window, self.threshold)
                                for window in sample
                            ]
                        )

                        self.eeg_samples.append(
                            (sample, adj_mat_sample, labels[sessionn][trial])
                        )

        self.eeg_samples = torch.tensor(self.eeg_samples)

        if need_save and path_to_save is not None:
            torch.save(self.eeg_samples, path_to_save)

    def _build_adjacency_matrix(window_data, threshold=0.7):
        # window_data: (62, window_size)
        correlations = torch.corrcoef(window_data)  # (62, 62)
        adjacency = (correlations > threshold).float()
        return adjacency

    def __len__(self):
        return len(self.eeg_samples)

    def __getitem__(self, idx):
        self.eeg_samples[idx]
