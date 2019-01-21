import argparse

import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as add_weight_norm
import torch.distributions as distributions
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np

class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))

class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J

class NICE(nn.Module):
    def __init__(self, prior, coupling, 
        in_out_dim, mid_dim=1000, hidden=5, mask_config=1.):
        """Initialize a NICE.

        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(NICE, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=in_out_dim, 
                     mid_dim=mid_dim, 
                     hidden=hidden, 
                     mask_config=(mask_config+i)%2) \
            for i in range(coupling)])
        self.scaling = Scaling(in_out_dim)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        return self.scaling(x)

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).cuda()
        return self.g(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)

class StandardLogistic(distributions.Distribution):
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
        z = distributions.Uniform(0., 1.).sample(size)
        return torch.log(z) - torch.log(1. - z)

def dequantize(x, reverse=False):
    '''Dequantize data.

    Args:
        x: input tensor.
        reverse: True in inference mode, False in training mode.
    Returns:
        dequantized data.
    '''
    if reverse:
        [B, _] = list(x.size())
        x = x.reshape((B, 1, 28, 28))
        return x
    else:
        [B, _, _, _] = list(x.size())
        x = x.reshape((B, 28*28))
        
        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B*28*28, ))
        x = (x * 255. + noise.reshape((B, 28*28))) / 256.
        return x

def main(args):
    device = torch.device("cuda:0")

    # model hyperparameters
    batch_size = args.batch_size
    latent = args.latent
    coupling = args.coupling
    mid_dim = args.mid_dim
    hidden = args.hidden
    mask_config = args.mask_config

    filename = 'bs%d_' % batch_size \
             + '%s_' % latent \
             + 'cp%d_' % coupling \
             + 'md%d_' % mid_dim \
             + 'hd%d_' % hidden

    # optimization hyperparameters
    lr = args.lr
    momentum = args.momentum
    decay = args.decay

    transform = torchvision.transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='~/torch/data',
        train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=batch_size, shuffle=True, num_workers=2)

    full_dim = 28 * 28
     
    if latent == 'normal':
        prior = distributions.Normal(
            torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    elif latent == 'logistic':
        prior = StandardLogistic()

    flow = NICE(prior=prior, 
                coupling=4, 
                in_out_dim=full_dim, 
                mid_dim=1000, 
                hidden=5).to(device)
    optimizer = optim.Adam(flow.parameters(), lr=lr, betas=(momentum, decay), eps=1e-4)

    total_iter = 0
    train = True
    running_loss = 0

    while train:
        for _, data in enumerate(trainloader, 1):
            flow.train()    # set to training mode
            if total_iter == args.max_iter:
                train = False
                break

            total_iter += 1
            optimizer.zero_grad()    # clear gradient tensors

            inputs, _ = data
            inputs = dequantize(inputs).to(device)

            # log-likelihood of input minibatch
            loss = -flow(inputs).mean()
            running_loss += float(loss)

            # backprop and update parameters
            loss.backward()
            optimizer.step()

            if total_iter % 1000 == 0:
                mean_loss = running_loss / 1000
                bit_per_dim = (mean_loss + np.log(256.) * full_dim) \
                            / (full_dim * np.log(2.))
                print('iter %s:' % total_iter, 
                    'loss = %.3f' % mean_loss, 
                    'bits/dim = %.3f' % bit_per_dim)
                running_loss = 0.0

                flow.eval()        # set to inference mode
                with torch.no_grad():
                    z, _ = flow.f(inputs)
                    reconst = flow.g(z)
                    reconst = dequantize(reconst, reverse=True)
                    samples = flow.sample(args.sample_size)
                    samples = dequantize(samples, reverse=True)
                    utils.save_image(utils.make_grid(reconst),
                        './reconstruction/' + filename +'iter%d.png' % total_iter)
                    utils.save_image(utils.make_grid(samples),
                        './samples/' + filename +'iter%d.png' % total_iter)

    print('Finished training!')

    torch.save({
        'total_iter': total_iter,
        'model_state_dict': flow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': batch_size, 
        'latent': latent, 
        'coupling': coupling, 
        'mid_dim': mid_dim, 
        'hidden': hidden, 
        'mask_config': mask_config}, 
        './models/mnist/' + filename +'iter%d.tar' % total_iter)

    print('Checkpoint Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MNIST NICE PyTorch implementation')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=200)
    parser.add_argument('--latent',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--coupling',
                        help='number of coupling layers.',
                        type=int,
                        default=4)
    parser.add_argument('--mid_dim',
                        help='number of units in a hidden layer.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='number of hidden layers in a coupling layer.',
                        type=int,
                        default=5)
    parser.add_argument('--mask_config',
                        help='mask configuration.',
                        type=float,
                        default=1.)
    parser.add_argument('--max_iter',
                        help='maximum number of iterations.',
                        type=int,
                        default=25000)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--momentum',
                        help='beta1 in Adam optimizer.',
                        type=float,
                        default=0.9)
    parser.add_argument('--decay',
                        help='beta2 in Adam optimizer.',
                        type=float,
                        default=0.999)
    args = parser.parse_args()
    main(args)
