import functools
import torch
import torch.nn as nn

from .modules import FreqNorm

ACTIVATIONS = dict({
    'linear': nn.Identity,
    'relu': nn.ReLU,
    'leakyrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2),
    'softmax': functools.partial(nn.Softmax, dim=-1), # softmax need to specify dim now
    'softplus': nn.Softplus,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
})

NORMALIZATIONS = dict({
    'layer': nn.LayerNorm,
    'batch': nn.BatchNorm1d,
    'freq': FreqNorm,
})