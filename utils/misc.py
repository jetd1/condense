# By Jet 2024
import math
import torch
from torch import nn, Tensor


def calc_pad(size, patch_size):
    new_size = math.ceil(size / patch_size) * patch_size
    pad_size = new_size - size
    pad_size_left = pad_size // 2
    pad_size_right = pad_size - pad_size_left
    return pad_size_left, pad_size_right


class ScalableUpsampling2D(nn.Upsample):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: Tensor):
        assert inputs.ndim == 4

        size_max = 2**31 - 2
        b, c, h, w = inputs.size()

        with torch.no_grad():
            out_h, out_w = super().forward(inputs.new_empty(1, 1, h, w)).shape[2:]

        single_data_size = c * max(out_h * out_w, h * w)

        if single_data_size > size_max:
            raise NotImplementedError("Size too big for upscaling.")

        if b * single_data_size <= size_max:
            return super().forward(inputs)

        max_b = int(math.floor(size_max / single_data_size))
        outputs = inputs.new_empty(b, c, out_h, out_w)
        n_mini_batch = math.ceil(b / max_b)
        for i in range(n_mini_batch):
            start = i * max_b
            end = start + max_b
            if end > b:
                end = b
            mini_batch_outputs = super().forward(inputs[start:end])
            outputs[start:end] = mini_batch_outputs

        return outputs
