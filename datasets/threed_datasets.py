import cv2
import json
import random
from utils.OS_data_utils import random_rotate_z, normalize_pc, augment_pc
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import logging
import copy
from torch.utils.data.distributed import DistributedSampler


class Four(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.phase = phase
        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_knn_negative_sample:
            self.negative_sample_num = config.dataset.negative_sample_num
            self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
            self.uid_to_index = {}
            for i, item in enumerate(self.split):
                self.uid_to_index[item['id']] = i
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_objaverse(self, meta):
        uid = meta["id"]
        data = np.load(meta['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        texts = []
        if 'text' in self.text_source:
            if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                texts.append(data["text"][0])
                text_feat.append(data["text_feat"][0][self.text_embed_version])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption"]) > 0:
                    texts.append(data["blip_caption"])
                    text_feat.append(data["blip_caption_feat"][self.text_embed_version])
            else:
                if len(data["msft_caption"]) > 0:
                    texts.append(data["msft_caption"])
                    text_feat.append(data["msft_caption_feat"][self.text_embed_version])

        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                idx = np.random.randint(len(data["retrieval_text"]))
                texts.append(data["retrieval_text"][idx])
                text_feat.append(
                    data["retrieval_text_feat"][idx]["original"])  # no prompt engineering for retrieval text

        if len(text_feat) > 0:
            assert len(text_feat) == len(texts)
            text_idx = np.random.randint(len(texts))
            text_feat = text_feat[text_idx]
            texts = texts[text_idx]
            text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
        else:
            text_feat = None
            texts = None

        if np.random.rand() < 0.5:
            img_feat = data['thumbnail_feat']
        else:
            idx = np.random.randint(data['image_feat'].shape[0])
            img_feat = data["image_feat"][idx]

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "text_feat": text_feat,
            "dataset": "Objaverse",
            "group": meta["group"],
            "name": uid,
            "texts": texts,
            "has_text": text_feat is not None,
        }

    def get_others(self, meta):
        data = np.load(meta['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        texts = []
        if 'text' in self.text_source:
            idx = np.random.randint(len(data["text"]))
            texts.append(data["text"][idx])
            text_feat.append(data["text_feat"][idx][self.text_embed_version])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption"]) > 0:
                    texts.append(data["blip_caption"])
                    text_feat.append(data["blip_caption_feat"][self.text_embed_version])
            else:
                if len(data["msft_caption"]) > 0:
                    texts.append(data["msft_caption"])
                    text_feat.append(data["msft_caption_feat"][self.text_embed_version])

        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                idx = np.random.randint(len(data["retrieval_text"]))
                texts.append(data["retrieval_text"][idx])
                text_feat.append(
                    data["retrieval_text_feat"][idx]["original"])  # no prompt engineering for retrieval text

        if len(text_feat) > 0:
            assert len(text_feat) == len(texts)
            text_idx = np.random.randint(len(texts))
            text_feat = text_feat[text_idx]
            texts = texts[text_idx]
            text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
        else:
            text_feat = None
            texts = None

        idx = np.random.randint(data['image_feat'].shape[0])
        img_feat = data["image_feat"][idx]

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "text_feat": text_feat,
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name": meta["id"],
            "texts": texts,
            "has_text": text_feat is not None,
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            if self.split[index]['dataset'] == "Objaverse":
                return self.get_objaverse(self.split[index])
            else:
                return self.get_others(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist()
            for i in knn_idx:
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                if self.split[idx]['dataset'] == "Objaverse":
                    data_list.append(self.get_objaverse(self.split[idx]))
                else:
                    data_list.append(self.get_others(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num


def minkowski_collate_fn(list_data):
    if isinstance(list_data[0], list):
        merged_list = []
        for data in list_data:
            merged_list += data
        list_data = merged_list
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "img_feat": [data["img_feat"] for data in list_data],
        "text_feat": [data["text_feat"] for data in list_data if data["text_feat"] is not None],
        "dataset": [data["dataset"] for data in list_data],
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "texts": [data["texts"] for data in list_data],
        "has_text_idx": [i for i, data in enumerate(list_data) if data["text_feat"] is not None],
    }


def make(config, phase, rank, world_size):
    if config.dataset.name == "Four":
        dataset = Four(config, phase, )
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=minkowski_collate_fn,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )
    else:
        raise NotImplementedError("Dataset %s not supported." % config.dataset.name)
    return data_loader


class ModelNet40Test(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.modelnet40.test_split, "r"))
        self.pcs = np.load(config.modelnet40.test_pc, allow_pickle=True)
        self.num_points = config.modelnet40.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.modelnet40.y_up
        clip_feat = np.load(config.modelnet40.clip_feat_path, allow_pickle=True).item()
        self.categories = list(clip_feat.keys())
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

        logging.info("ModelNet40Test: %d samples" % len(self.split))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        pc = copy.deepcopy(self.pcs[index])
        n = pc['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = pc['xyz'][idx]
        rgb = pc['rgb'][idx]
        rgb = rgb / 255.0  # 100, scale to 0.4 to make it consistent with the training data
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]

        xyz = normalize_pc(xyz)

        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": self.split[index]["name"],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)


def minkowski_modelnet40_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
    }


def make_modelnet40test(config):
    dataset = ModelNet40Test(config)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.modelnet40.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.modelnet40.test_batch_size, \
        pin_memory=True, \
        shuffle=False
    )
    return data_loader


class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)

        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        data = np.load(self.split[index]['data_path'], allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "group": self.split[index]['group'],
            "name": self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)


def minkowski_objaverse_lvis_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
    }


def make_objaverse_lvis(config):
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=minkowski_objaverse_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory=True, \
        shuffle=False
    )


class ScanObjectNNTest(Dataset):
    def __init__(self, config):
        self.data = np.load(config.scanobjectnn.data_path, allow_pickle=True).item()
        self.num_points = config.scanobjectnn.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.scanobjectnn.y_up
        clip_feat = np.load(config.scanobjectnn.clip_feat_path, allow_pickle=True).item()
        self.categories = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed",
                           "pillow", "sink", "sofa", "toilet"]
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
            self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)
        logging.info("ScanObjectNNTest: %d samples" % self.__len__())
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        xyz = copy.deepcopy(self.data['xyz'][index])
        if 'rgb' not in self.data:
            rgb = np.ones_like(xyz) * 0.4
        else:
            rgb = self.data['rgb'][index]
        label = self.data['label'][index]
        n = xyz.shape[0]
        if n != self.num_points:
            idx = np.random.choice(n, self.num_points)  # random.sample(range(n), self.num_points)
            xyz = xyz[idx]
            rgb = rgb[idx]
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]

        xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": str(index),
            "category": label,
        }

    def __len__(self):
        return len(self.data['xyz'])


def make_scanobjectnntest(config):
    dataset = ScanObjectNNTest(config)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.scanobjectnn.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.scanobjectnn.test_batch_size, \
        pin_memory=True, \
        shuffle=False
    )
    return data_loader

import glob
import os
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(
        self,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root="data/s3dis",
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(S3DISDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        self.data_list = self.get_data_list()


    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            data = torch.load(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "ponder" + data_name.replace(os.path.sep, "-")
            data = dict(cache_name)
        name = (
            os.path.basename(self.data_list[idx % len(self.data_list)])
            .split("_")[0]
            .replace("R", " r")
        )
        coord = data["coord"]
        color = data["color"]
        scene_id = data_path
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


class S3DISRGBDDataset(S3DISDataset):
    def __init__(
        self,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root="data/s3dis",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        num_cameras=5,
        render_semantic=True,
        six_fold=False,
        loop=1,
    ):
        super(S3DISRGBDDataset, self).__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            cache=cache,
            loop=loop,
        )
        self.num_cameras = num_cameras
        self.render_semantic = render_semantic
        self.six_fold = six_fold

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError

        print("Filtering S3DIS RGBD dataset...")
        filtered_data_list = []
        for data_path in data_list:
            rgbd_paths = glob.glob(
                os.path.join(data_path.split(".pth")[0] + "_rgbd", "*.pth")
            )
            if len(rgbd_paths) <= 0:
                # print(f"{data_path} has no rgbd data.")
                continue
            filtered_data_list.append(data_path)
        print(
            f"Finish filtering! Totally {len(filtered_data_list)} from {len(data_list)} data."
        )
        return filtered_data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            data = torch.load(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "ponder" + data_name.replace(os.path.sep, "-")
            data = dict(cache_name)

        rgbd_paths = glob.glob(
            os.path.join(data_path.split(".pth")[0] + "_rgbd", "*.pth")
        )

        if len(rgbd_paths) <= 0:
            print(f"{data_path} has no rgbd data.")
            return self.get_data(np.random.randint(0, self.__len__()))

        rgbd_paths = np.random.choice(
            rgbd_paths, self.num_cameras, replace=self.num_cameras > len(rgbd_paths)
        )
        rgbd_dicts = [torch.load(p) for p in rgbd_paths]

        for i in range(len(rgbd_dicts)):
            if (rgbd_dicts[i]["depth_mask"]).mean() < 0.25:
                os.rename(rgbd_paths[i], rgbd_paths[i] + ".bad")
                return self.get_data(idx)

        name = (
            os.path.basename(self.data_list[idx % len(self.data_list)])
            .split("_")[0]
            .replace("R", " r")
        )
        coord = data["coord"]
        color = data["color"]
        scene_id = data_path
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
            intrinsic=np.stack([d["intrinsic"] for d in rgbd_dicts], axis=0).astype(
                np.float32
            ),
            extrinsic=np.stack(
                [np.linalg.inv(d["extrinsic"]) for d in rgbd_dicts], axis=0
            ).astype(np.float32),
            rgb=np.stack([d["rgb"].astype(np.float32) for d in rgbd_dicts], axis=0),
            depth=np.stack(
                [
                    d["depth"].astype(np.float32)
                    * d["depth_mask"].astype(np.float32)
                    * (d["depth"] < 65535).astype(np.float32)
                    for d in rgbd_dicts
                ],
                axis=0,
            ),
            depth_scale=1.0 / 4000.0,
        )

        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
        if self.render_semantic:
            for d in rgbd_dicts:
                d["semantic_map"][d["semantic_map"] <= 0] = -1
                d["semantic_map"][d["semantic_map"] > 40] = -1
                d["semantic_map"] = d["semantic_map"].astype(np.int16)
            data_dict.update(
                dict(semantic=np.stack([d["semantic_map"] for d in rgbd_dicts], axis=0))
            )
        if (
            self.six_fold
        ):  # pretrain for 6-fold cross validation, ignore semantic labels to avoid leaking information
            data_dict["semantic"] = np.zeros_like(data_dict["semantic"]) - 1
        return data_dict


class ScanNetDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(ScanNetDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if lr_file:
            self.data_list = [
                os.path.join(data_root, "train", name + ".pth")
                for name in np.loadtxt(lr_file, dtype=str)
            ]
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            data = torch.load(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "ponder" + data_name.replace(os.path.sep, "-")
            data = dict(cache_name)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt20" in data.keys():
            segment = data["semantic_gt20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


class ScanNet200Dataset(ScanNetDataset):
    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt200" in data.keys():
            segment = data["semantic_gt200"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            segment[sampled_index] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict


class ScanNetRGBDDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        rgbd_root="data/scannet/rgbd",
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        frame_interval=10,
        nearby_num=2,
        nearby_interval=20,
        num_cameras=5,
        render_semantic=True,
        align_axis=False,
        loop=1,
    ):
        super(ScanNetRGBDDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.rgbd_root = rgbd_root
        self.frame_interval = frame_interval
        self.nearby_num = nearby_num
        self.nearby_interval = nearby_interval
        self.num_cameras = num_cameras
        self.render_semantic = render_semantic
        self.align_axis = align_axis

        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if lr_file:
            full_data_list = self.get_data_list()
            self.data_list = []
            lr_list = np.loadtxt(lr_file, dtype=str)
            for data_dict in full_data_list:
                if data_dict["scene"] in lr_list:
                    self.data_list.append(data_dict)
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index

    def get_data_list(self):
        self.axis_align_matrix_list = {}
        self.intrinsic_list = {}
        self.frame_lists = {}

        # Get all models
        data_list = []
        split_json = os.path.join(os.path.join(self.data_root, self.split + ".json"))

        if os.path.exists(split_json):
            with open(split_json, "r") as f:
                data_list = json.load(f)
        else:
            scene_list = [
                filename.split(".")[0]
                for filename in os.listdir(os.path.join(self.data_root, self.split))
            ]

            skip_list = []
            skip_counter = 0
            skip_file = os.path.join(os.path.join(self.data_root, "skip.lst"))
            if os.path.exists(skip_file):
                with open(skip_file, "r") as f:
                    for i in f.read().split("\n"):
                        if not i:
                            continue
                        scene_name, frame_idx = i.split()
                        skip_list.append((scene_name, int(frame_idx)))

            # walk through the subfolder
            from tqdm import tqdm

            for scene_name in tqdm(scene_list):
                # filenames = os.listdir(os.path.join(subpath, m, 'pointcloud'))
                frame_list = self.get_frame_list(scene_name)

                # for test and val, we only use 1/10 of the data, since those data will not affect
                # the training and we use them just for visualization and debugging
                if self.split == "val":
                    frame_list = frame_list[::10]
                if self.split == "test":
                    frame_list = frame_list[::10]

                for frame_idx in frame_list[
                    self.nearby_num
                    * self.nearby_interval : -(self.nearby_num + 1)
                    * self.nearby_interval : self.frame_interval
                ]:
                    frame_idx = int(frame_idx.split(".")[0])
                    if (scene_name, frame_idx) in skip_list:
                        skip_counter += 1
                        continue
                    data_list.append({"scene": scene_name, "frame": frame_idx})

            self.logger.info(
                f"ScanNet: <{skip_counter} Frames will be skipped in {self.split} data.>"
            )

            with open(split_json, "w") as f:
                json.dump(data_list, f)

        data_dict = dict(list)
        for data in data_list:
            data_dict[data["scene"]].append(data["frame"])

        data_list = []
        for scene_name, frame_list in data_dict.items():
            data_list.append({"scene": scene_name, "frame": frame_list})

        return data_list

    def get_data(self, idx):
        scene_name = self.data_list[idx % len(self.data_list)]["scene"]
        frame_list = self.data_list[idx % len(self.data_list)]["frame"]
        scene_path = os.path.join(self.data_root, self.split, f"{scene_name}.pth")
        if not self.cache:
            data = torch.load(scene_path)
        else:
            data_name = scene_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "ponder" + data_name.replace(os.path.sep, "-")
            data = dict(cache_name)

        if self.num_cameras > len(frame_list):
            print(
                f"Warning: {scene_name} has only {len(frame_list)} frames, "
                f"but {self.num_cameras} cameras are required."
            )
        frame_idxs = np.random.choice(
            frame_list, self.num_cameras, replace=self.num_cameras > len(frame_list)
        )
        intrinsic, extrinsic, rgb, depth = (
            [],
            [],
            [],
            [],
        )

        if self.render_semantic:
            semantic = []
        for frame_idx in frame_idxs:
            if not self.render_semantic:
                intri, rot, transl, rgb_im, depth_im = self.get_2d_meta(
                    scene_name, frame_idx
                )
            else:
                intri, rot, transl, rgb_im, depth_im, semantic_im = self.get_2d_meta(
                    scene_name, frame_idx
                )
                assert semantic_im.max() <= 20, semantic_im
                semantic.append(semantic_im)
            intrinsic.append(intri)
            extri = np.eye(4)
            extri[:3, :3] = rot
            extri[:3, 3] = transl
            extrinsic.append(extri)
            rgb.append(rgb_im)
            depth.append(depth_im)

        intrinsic = np.stack(intrinsic, axis=0)
        extrinsic = np.stack(extrinsic, axis=0)
        rgb = np.stack(rgb, axis=0)
        depth = np.stack(depth, axis=0)

        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt20" in data.keys():
            segment = data["semantic_gt20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            rgb=rgb,
            depth=depth,
            depth_scale=1.0 / 1000.0,
            id=f"{scene_name}/{frame_idxs[0]}",
        )
        if self.render_semantic:
            semantic = np.stack(semantic, axis=0)
            data_dict.update(dict(semantic=semantic))

        if self.la:
            sampled_index = self.la[self.get_data_name(scene_path)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
            data_dict["semantic"] = np.zeros_like(data_dict["semantic"]) - 1

        return data_dict

    def get_data_name(self, scene_path):
        return os.path.basename(scene_path).split(".")[0]

    def get_frame_list(self, scene_name):
        if scene_name in self.frame_lists:
            return self.frame_lists[scene_name]

        if not os.path.exists(os.path.join(self.rgbd_root, scene_name, "color")):
            return []

        frame_list = os.listdir(os.path.join(self.rgbd_root, scene_name, "color"))
        frame_list = list(frame_list)
        frame_list = [frame for frame in frame_list if frame.endswith(".jpg")]
        frame_list.sort(key=lambda x: int(x.split(".")[0]))
        self.frame_lists[scene_name] = frame_list
        return self.frame_lists[scene_name]

    def get_axis_align_matrix(self, scene_name):
        if scene_name in self.axis_align_matrix_list:
            return self.axis_align_matrix_list[scene_name]
        txt_file = os.path.join(self.rgbd_root, scene_name, "%s.txt" % scene_name)
        # align axis
        with open(txt_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if "axisAlignment" in line:
                self.axis_align_matrix_list[scene_name] = [
                    float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
                ]
                break
        self.axis_align_matrix_list[scene_name] = np.array(
            self.axis_align_matrix_list[scene_name]
        ).reshape((4, 4))
        return self.axis_align_matrix_list[scene_name]

    def get_intrinsic(self, scene_name):
        if scene_name in self.intrinsic_list:
            return self.intrinsic_list[scene_name]
        self.intrinsic_list[scene_name] = np.loadtxt(
            os.path.join(self.rgbd_root, scene_name, "intrinsic", "intrinsic_depth.txt")
        )
        return self.intrinsic_list[scene_name]

    def get_2d_meta(self, scene_name, frame_idx):
        # framelist
        frame_list = self.get_frame_list(scene_name)
        intrinsic = self.get_intrinsic(scene_name)
        if self.align_axis:
            axis_align_matrix = self.get_axis_align_matrix(scene_name)

        if not self.render_semantic:
            rgb_im, depth_im, pose = self.read_data(scene_name, frame_list[frame_idx])
        else:
            rgb_im, depth_im, pose, semantic_im = self.read_data(
                scene_name, frame_list[frame_idx]
            )
            semantic_im_40 = cv2.resize(
                semantic_im,
                (depth_im.shape[1], depth_im.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            semantic_im_40 = semantic_im_40.astype(np.int16)
            semantic_im = np.zeros_like(semantic_im_40) - 1

        rgb_im = cv2.resize(rgb_im, (depth_im.shape[1], depth_im.shape[0]))
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)  # H, W, 3
        depth_im = depth_im.astype(np.float32)  # H, W

        if self.align_axis:
            pose = np.matmul(axis_align_matrix, pose)
        pose = np.linalg.inv(pose)

        intrinsic = np.array(intrinsic)
        rotation = np.array(pose)[:3, :3]
        translation = np.array(pose)[:3, 3]

        if not self.render_semantic:
            return intrinsic, rotation, translation, rgb_im, depth_im
        else:
            return intrinsic, rotation, translation, rgb_im, depth_im, semantic_im

    def read_data(self, scene_name, frame_name):
        color_path = os.path.join(self.rgbd_root, scene_name, "color", frame_name)
        depth_path = os.path.join(
            self.rgbd_root, scene_name, "depth", frame_name.replace(".jpg", ".png")
        )

        depth_im = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb_im = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)

        pose = np.loadtxt(
            os.path.join(
                self.rgbd_root,
                scene_name,
                "pose",
                frame_name.replace(".jpg", ".txt"),
            )
        )

        if not self.render_semantic:
            return rgb_im, depth_im, pose
        else:
            seg_path = os.path.join(
                self.rgbd_root,
                scene_name,
                "label",
                frame_name.replace(".jpg", ".png"),
            )
            semantic_im = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            return rgb_im, depth_im, pose, semantic_im

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop