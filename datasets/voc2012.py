# By Jet 2024
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VOCTestDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root = config.dataset.root

        self.img_dir = os.path.join(self.root, 'VOCdevkit/VOC2012/JPEGImages')
        self.ann_dir = os.path.join(self.root, 'VOCdevkit/VOC2012/SegmentationClass')

        split_file = os.path.join(self.root, 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        with open(split_file, 'r') as f:
            self.img_ids = f.read().splitlines()

        self.num_classes = 21
        self.size_minimum = config.segmentation.test_size_minimum
        self.size_maximum = config.segmentation.test_size_maximum

    def __len__(self):
        if self.config.debug:
            return 100
        return len(self.img_ids)

    @staticmethod
    def img_transform(img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())
        return img

    @staticmethod
    def segm_transform(segm):
        # to tensor, -1 to 20
        segm = torch.from_numpy(np.array(segm)).long()
        segm[segm == 255] = -1
        return segm

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        ann_path = os.path.join(self.ann_dir, img_id + '.png')

        img_ori = Image.open(img_path).convert('RGB')
        segm = Image.open(ann_path)
        assert img_ori.size == segm.size
        img = img_ori.copy()

        w, h = img.size
        if w < self.size_minimum or h < self.size_minimum:
            resize_scale = self.size_minimum / w if w < h else self.size_minimum / h
            img = img.resize(
                (int(resize_scale * w), int(resize_scale * h)),
                resample=Image.Resampling.BILINEAR
            )
        elif w > self.size_maximum or h > self.size_maximum:
            resize_scale = self.size_maximum / w if w > h else self.size_maximum / h
            img = img.resize(
                (int(resize_scale * w), int(resize_scale * h)),
                resample=Image.Resampling.BILINEAR
            )

        img = self.img_transform(img)
        segm = self.segm_transform(segm)

        output = dict()
        output['img_ori'] = np.array(img_ori)
        output['img'] = img.contiguous()
        output['label'] = segm.contiguous()
        return output
