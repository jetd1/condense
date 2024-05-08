import os
import numpy as np
from torch.utils.data import Dataset


class Places205TestDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_root = config.dataset.root
        data = np.load(os.path.join(self.data_root, 'places205_val.npz'))
        self.images = data['data'].astype(np.float32) / 255
        self.labels = data['label']
        print(f'# {len(self.images)} test images loaded')

    def __len__(self):
        if self.config.debug:
            return 100
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'img': self.images[idx],
            'label': self.labels[idx]
        }
