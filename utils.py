"""Utility functions for NICE.
"""

import torch
import torch.nn.functional as F

def dequantize(x, dataset):
    '''Dequantize data.

    Add noise sampled from Uniform(0, 1) to each pixel (in [0, 255]).

    Args:
        x: input tensor.
        reverse: True in inference mode, False in training mode.
    Returns:
        dequantized data.
    '''
    noise = torch.distributions.Uniform(0., 1.).sample(x.size())
    return (x * 255. + noise) / 256.

def prepare_data(x, dataset, zca=None, mean=None, reverse=False):
    """Prepares data for NICE.

    In training mode, flatten and dequantize the input.
    In inference mode, reshape tensor into image size.

    Args:
        x: input minibatch.
        dataset: name of dataset.
        zca: ZCA whitening transformation matrix.
        mean: center of original dataset.
        reverse: True if in inference mode, False if in training mode.
    Returns:
        transformed data.
    """
    if reverse:
        assert len(list(x.size())) == 2
        [B, W] = list(x.size())

        if dataset in ['mnist', 'fashion-mnist']:
            assert W == 1 * 28 * 28
            x += mean
            x = x.reshape((B, 1, 28, 28))
        elif dataset in ['svhn', 'cifar10']:
            assert W == 3 * 32 * 32
            x = torch.matmul(x, zca.inverse()) + mean
            x = x.reshape((B, 3, 32, 32))
    else:
        assert len(list(x.size())) == 4
        [B, C, H, W] = list(x.size())

        if dataset in ['mnist', 'fashion-mnist']:
            assert [C, H, W] == [1, 28, 28]
        elif dataset in ['svhn', 'cifar10']:
            assert [C, H, W] == [3, 32, 32]

        x = dequantize(x, dataset)
        x = x.reshape((B, C*H*W))

        if dataset in ['mnist', 'fashion-mnist']:
            x -= mean
        elif dataset in ['svhn', 'cifar10']:
            x = torch.matmul((x - mean), zca)
    return x

"""Standard logistic distribution.
"""
class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__()

    def log_prob(self, x):
        """Computes data log-likelihood.

        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        """Samples from the distribution.

        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(0., 1.).sample(size).cuda()
        return torch.log(z) - torch.log(1. - z)