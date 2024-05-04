# By Jet 2024
import torch
import torchvision.transforms
from backbone2.base import Backbone2Base


MODEL_INFO = {
    'dinov2_vits14': {'n_channels': 384, 'alignment': 14},
    'dinov2_vitb14': {'n_channels': 768, 'alignment': 14},
    'dinov2_vitl14': {'n_channels': 1024, 'alignment': 14},
    'dinov2_vitg14': {'n_channels': 1536, 'alignment': 14},
    'dinov2_vits14_reg': {'n_channels': 384, 'alignment': 14},
    'dinov2_vitb14_reg': {'n_channels': 768, 'alignment': 14},
    'dinov2_vitl14_reg': {'n_channels': 1024, 'alignment': 14},
    'dinov2_vitg14_reg': {'n_channels': 1536, 'alignment': 14},
}


class DINOv2(Backbone2Base):
    def __init__(self, model_name: str, pretrained=True, feature_type='patch'):
        super().__init__()

        assert model_name in MODEL_INFO
        assert feature_type in ['patch', 'global']
        # 'patch': patch tokens, 'global': cls concat w/ avgpooled patch tokens

        self.model_name = model_name
        self.feature_type = feature_type

        # Load Model
        self.model = None
        self.preprocessor = None

        self._load_model(model_name, pretrained)
        self._load_preprocessor()

        info = MODEL_INFO[model_name]

        if self.feature_type == 'patch':
            self._num_channels = info['n_channels']
        else:
            self._num_channels = info['n_channels'] * 2

        self._alignment = info['alignment']

    def _load_model(self, model_name: str, pretrained=True):
        self.model = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=pretrained)

    def _load_preprocessor(self):
        self.preprocessor = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

    def get_num_channels(self):
        return self._num_channels

    def get_alignment(self):
        return self._alignment

    def get_feature(self, x):
        x = self.preprocessor(x)

        if self.feature_type == 'patch':
            bs, c, h, w = x.shape
            ret = self.model.forward_features(x)['x_norm_patchtokens']
            ret = ret.reshape(bs, h // self._alignment, w // self._alignment, self._num_channels)
        else:
            features = self.model.forward_features(x)
            ret = torch.cat(
                (features['x_norm_clstoken'], features['x_norm_patchtokens'].mean(dim=1)), dim=1)

        return ret
