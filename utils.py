import torch
import glob
from PIL import Image
from torch import nn
import os
import numpy as np
from scipy import linalg
import torchvision
from torchvision import transforms

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label=None, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.transform = transform
        self.target_transform = target_transform
        self.label = label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        label = self.label
            
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    # print('calculate diff')
    diff = mu1 - mu2

    # Product might be almost singular
    # print('calculate covmean')
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=2e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # print('calculate tr_covmean')
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def mnist_dataset():
    transform_mnist=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])]
    )
    # target_transform = transforms.Lambda(lambda y: 0)
    target_transform = None
    mnist_train = torchvision.datasets.MNIST('your/dataroot', train=True, download=False, transform=transform_mnist, target_transform=target_transform)
    return mnist_train

def reverse_mnist_dataset():
    transform_mnist_reverse=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: 1-x),
        transforms.Normalize([0.5], [0.5])]
    )
    target_transform = transforms.Lambda(lambda y: 1)
    mnist_train = torchvision.datasets.MNIST('your/dataroot', train=True, download=False, transform=transform_mnist_reverse, target_transform=target_transform)
    return mnist_train

transform_cifar=transforms.Compose(
    [transforms.Resize((32, 32)), 
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(), 
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
     ]
)

def target_transform_cifar(target):
    return transforms.Lambda(lambda x: target)

def cifar_single_dataset(label, target=None):
    if target is None:
        target_transform = None
    else:
        target_transform = target_transform_cifar(target)

    cifar = torchvision.datasets.CIFAR10('your/dataroot', train=True, download=False, transform=transform_cifar, target_transform=target_transform)
    targets = cifar.targets

    digit_list = []
    for i in range(len(cifar)):
        if targets[i] == label:
            digit_list.append(i)
    digit_list = torch.tensor(digit_list)

    train_dataset = torch.utils.data.Subset(cifar, digit_list)
    return train_dataset