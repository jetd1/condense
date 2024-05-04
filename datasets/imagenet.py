import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageNetTestDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.image_dir = config.dataset.root
        with open(config.dataset.imagenet.test_ground_truth, 'r') as f:
            gt = f.read().split()
            self.labels = [int(x) for x in gt]
        try:
            max_samples = config.dataset.imagenet.test_max_samples
            assert max_samples > 0
            self.labels = self.labels[:max_samples]
        except AttributeError:
            pass

    def __len__(self):
        if self.config.debug:
            return 100
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, f"ILSVRC2012_val_{idx + 1:08d}.JPEG")
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.config.classification.test_image_size)
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        return {
            'img': img,
            'label': self.labels[idx]
        }
