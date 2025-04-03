from torch.utils.data import Dataset
from data_api import DataAPI


class SEEDIVDataset(Dataset):
    def __init__(self, path, group_shape):
        super().__init__()
        self.dataset = DataAPI(path, group_shape=group_shape)
        self.dataset_size = self.dataset.root[self.dataset.LABELS].shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self.dataset.get_sample(index)
      
