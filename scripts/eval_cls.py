import torch
from torch import nn
from backbone2.dinov2 import DINOv2
from utils.arg import get_config
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class ClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = DINOv2(config.backbone.name, pretrained=False, feature_type='global')
        try:
            self.backbone.model.load_state_dict(
                torch.load(config.backbone.ckpt), strict=True)
        except AttributeError:
            pass

        self.decode_head = nn.Linear(
            self.backbone.get_num_channels(),
            config.dataset.n_classes
        )
        try:
            self.decode_head.load_state_dict(
                torch.load(config.head.ckpt), strict=True)
        except AttributeError:
            pass

        self.backbone.requires_grad_(False)
        self.decode_head.requires_grad_(True)

        self.config = config

    def forward(self, x):
        with torch.no_grad():
            ret = self.backbone(x)
        ret = self.decode_head(ret)
        return ret


if __name__ == '__main__':
    cfg, _, _ = get_config()
    model = ClassificationModel(cfg)
    model.eval()
    model.cuda()
    torch.backends.cudnn.benchmark = True

    if cfg.dataset.name == 'imagenet':
        from datasets.imagenet import ImageNetTestDataset
        dataset = ImageNetTestDataset(cfg)
    elif cfg.dataset.name == 'places205':
        from datasets.places205 import Places205TestDataset
        dataset = Places205TestDataset(cfg)
    else:
        raise NotImplementedError(f"Unknown Dataset {cfg.dataset.name}")

    dataloader = DataLoader(
        dataset,
        batch_size=50,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        pin_memory=True
    )

    total = len(dataset)
    correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            data: dict
            inputs = data['img'].cuda()
            labels = data['label']
            outputs = model(inputs)
            pred = outputs.argmax(1).cpu().long()

            # Image.fromarray((inputs.cpu().numpy().squeeze(0) * 255).astype(np.uint8).transpose((1, 2, 0))).show()
            # print(f'pred: {pred.item()}, label: {labels.item()}')
            correct += (pred == labels).sum().item()

    print(f"Accuracy: {correct / total:.5f}")
