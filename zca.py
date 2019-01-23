import torch
import torchvision
import numpy as np

def zca(x):
    """Computes ZCA transformation for the dataset.

    Args:
        x: dataset.
    Returns:
        ZCA transformation matrix and mean matrix.
    """
    [B, C, H, W] = list(x.size())
    x = x.reshape((B, C*H*W))       # flattern the data
    mean = torch.mean(x, dim=0, keepdim=True)
    x -= mean                
    covariance = torch.matmul(x.transpose(0, 1), x) / B
    U, S, V = np.linalg.svd(covariance.numpy())
    eps = 1e-3
    W = np.matmul(np.matmul(U, np.diag(1. / np.sqrt(S + eps))), U.T)
    return torch.tensor(W), mean

# whiten CIFAR10 
trainset = torchvision.datasets.CIFAR10(
    root='../../data', transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=50000, shuffle=False, num_workers=4)
for _, data in enumerate(trainloader):
    break
images, _ = data
W, mean = zca(images)
torch.save(W, './statistics/cifar10_zca_3.pt')
# torch.save(mean, 'cifar10_mean.pt')
samples = images[0:64]
out = torch.matmul(samples.reshape((64, 3*32*32)), W)
out = out.reshape((64, 3, 32, 32))
torchvision.utils.save_image(torchvision.utils.make_grid(out), 'cifar_zca_3.png')

# whiten SVHN
trainset = torchvision.datasets.SVHN(
    root='../../data', transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=73257, shuffle=False, num_workers=4)
for _, data in enumerate(trainloader):
    break
images, _ = data
W, mean = zca(images)
torch.save(W, './statistics/svhn_zca_3.pt')
# torch.save(mean, 'cifar10_mean.pt')
samples = images[0:64]
out = torch.matmul(samples.reshape((64, 3*32*32)), W)
out = out.reshape((64, 3, 32, 32))
torchvision.utils.save_image(torchvision.utils.make_grid(out), 'svhn_zca_3.png')