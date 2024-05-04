import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ADE20kTestDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.list_sample = [
            json.loads(x.rstrip())
            for x in open(config.dataset.ade20k.test_odgt, 'r')
        ]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# test samples: {}'.format(self.num_sample))
        self.root_dataset = config.dataset.root
        self.size_minimum = config.segmentation.test_size_minimum
        self.size_maximum = config.segmentation.test_size_maximum

    def __getitem__(self, index):
        this_record: dict = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img_ori = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert segm.mode == "L"
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
        # output['info'] = this_record['fpath_img']
        return output

    @staticmethod
    def img_transform(img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())
        return img

    @staticmethod
    def segm_transform(segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    def __len__(self):
        if self.config.debug:
            return 100
        return self.num_sample
