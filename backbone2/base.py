# By Jet 2024
import torch
import abc


class Backbone2Base(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    @abc.abstractmethod
    def get_feature(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_channels(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_alignment(self):
        raise NotImplementedError

    def forward(self, x):
        return self.get_feature(x)
