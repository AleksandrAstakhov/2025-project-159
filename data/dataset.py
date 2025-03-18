from torch.utils.data import Dataset

import scipy.io as sio
import torch


class SEEDIVDataset(Dataset):
    def __init__(
        self,
        data_files,
        session_idx,
        window_size,
        window_stride,
        windows_number,
        threshold,
    ):
        super().__init__()

        labels = [
            [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
        ]

        self.data_files = data_files
        self.session_idx = session_idx
        self.threshold = threshold
        self.window_size = window_size
        self.window_stride = window_stride
        self.windows_number = windows_number

        self.eeg_samples = []

        for mat_file in self.data_files:
            mat_dict = sio.loadmat(mat_file)

            for trial in range(1, 25):
                eeg_signal = torch.tensor(mat_dict[f"cz_eeg{trial}"])

                eeg_windows = eeg_signal.unfold(dimension=1, size=3, step=3).permute(
                    1, 0, 2
                )

                for i in range(len(eeg_windows) // self.windows_number):
                    start_idx = i * self.windows_number
                    end_idx = (i + 1) * self.window_size
                    sample = eeg_windows[start_idx:end_idx]
                    self.eeg_samples.append((sample, labels[self.session_idx][trial]))
                    
        
        self.eeg_samples = torch.tensor(self.eeg_samples)

    def __len__(self):
        return len(self.eeg_samples)

    def __getitem__(self, idx):
        self.eeg_samples[idx]
