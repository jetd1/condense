# By Jet 2024
import logging
import torch


class SingleLinear(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm='sync_bn',
                 upscaler=None):
        super().__init__()

        assert norm == 'sync_bn'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.upscaler = upscaler

        self.bn = torch.nn.SyncBatchNorm(self.in_channels)
        self.conv_seg = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        ret = self.conv_seg(self.bn(x))
        if self.upscaler is not None:
            logging.getLogger("user").debug(f"Upscaling from shape {ret.shape}.")
            ret = self.upscaler(ret)
        return ret
