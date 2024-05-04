import torch
from torch import nn
from backbone2.dinov2 import DINOv2
from head2.seg import SingleLinear
from utils.arg import get_config
from torch.utils.data.dataloader import DataLoader

from utils.eval import sliding_window_inference
from utils.metric import calc_i_and_u
import numpy as np
from tqdm import tqdm
import torch.nn.functional as nnf


class SegmentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = DINOv2(config.backbone.name, pretrained=False)
        try:
            self.backbone.model.load_state_dict(torch.load(config.backbone.ckpt), strict=True)
        except AttributeError:
            pass

        self.decode_head = SingleLinear(
            in_channels=self.backbone.get_num_channels(),
            out_channels=config.dataset.n_classes,
            upscaler=nn.Upsample(scale_factor=self.backbone.get_alignment(), mode='bilinear')
        )
        try:
            self.load_state_dict(torch.load(config.head.ckpt)["state_dict"], strict=False)
        except AttributeError:
            pass

        self.backbone.requires_grad_(False)
        self.decode_head.requires_grad_(True)

        self.config = config

    def forward(self, x):
        with torch.no_grad():
            ret = self.backbone(x)
        ret = ret.permute(0, 3, 1, 2)
        ret = self.decode_head(ret)
        return ret


if __name__ == '__main__':
    cfg, _, _ = get_config()
    model = SegmentationModel(cfg)
    model.eval()
    model.cuda()

    if cfg.dataset.name == 'ade20k':
        from datasets.ade20k import ADE20kTestDataset
        dataset = ADE20kTestDataset(cfg)
    elif cfg.dataset.name == 'voc2012':
        from datasets.voc2012 import VOCTestDataset
        dataset = VOCTestDataset(cfg)
    else:
        raise NotImplementedError(f"Unknown Dataset {cfg.dataset.name}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=True
    )

    inter = np.zeros((cfg.dataset.n_classes, len(dataset)))
    union = np.zeros((cfg.dataset.n_classes, len(dataset)))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            data: dict
            inputs = data['img'].cuda()
            labels = data['label']
            outputs = sliding_window_inference(
                inputs,
                model,
                crop_size=cfg.segmentation.test_slide_crop_size,
                stride=cfg.segmentation.test_slide_stride,
                n_out_channels=cfg.dataset.n_classes
            )
            outputs = nnf.interpolate(
                outputs,
                size=labels.shape[1:],
                mode='bilinear'
            )
            pred = outputs.argmax(1).cpu().long()
            # print(inputs.shape)
            # print(outputs.shape)
            # print(pred.shape)
            # print(labels.shape)
            # from utils.vis import render_segmentation
            # seg = render_segmentation(pred.numpy()[0], 'ade20k')
            # seg.show()
            # gt = render_segmentation(labels.numpy()[0], 'ade20k')
            # gt.show()

            inter[:, i], union[:, i] = calc_i_and_u(pred, labels, cfg.dataset.n_classes)

    # TODO: This is wrong for small scale eval (legacy).
    IoU = 1.0 * np.sum(inter, axis=1) / np.sum(np.spacing(1) + union, axis=1)
    mIoU = IoU.sum() / (IoU != 0).sum()
    print(IoU)
    print(mIoU)
