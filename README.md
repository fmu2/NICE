# NICE
_A PyTorch implementation of the training procedure of [NICE: Nonlinear Independent Components Estimation](https://arxiv.org/pdf/1410.8516.pdf)_. The original implementation in theano and pylearn2 can be found at <https://github.com/laurent-dinh/nice>. 

## Imlementation Details
This implementation supports training on four datasets, namely **MNIST**, **Fashion-MNIST**, **SVHN** and **CIFAR-10**. For each dataset, only the training split is used for learning the distribution. Labels are left untouched. As is common practice, data is centered at zero before further processing. As suggested by the authors, random noise is added to dequantize the data. For SVHN and CIFAR-10, ZCA is performed to whiten the data. The means and ZCA transformation matrices were computed offline (see .pt files in ./statistics). The same set of hyperparameters (e.g. number of coupling layers, number of hidden layers in a coupling layer, number of hidden units in a hidden layer, etc.) as the paper suggests is used. Adam with default parameters are used for optimization. Samples (see below) generated from the learned distribution resemble those shown in the paper.

**Note: For SVHN and CIFAR-10, add log-determinant from ZCA transformation to data log-likelihood.**

## Samples
The samples are generated from models trained with default parameters.

**MNIST**

![](https://github.com/fmu2/NICE/blob/master/samples/mnist_bs200_logistic_cp4_md1000_hd5_iter1000.png "MNIST 1000 iterations")

![](https://github.com/fmu2/NICE/blob/master/samples/mnist_bs200_logistic_cp4_md1000_hd5_iter25000.png "MNIST 25000 iterations")

**Fashion-MNIST**






