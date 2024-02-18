import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import random
from spectral_norm import spectral_norm_flex
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.tensorboard import SummaryWriter

# from mlp_mixer_pytorch import MLPMixer

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--d_depth', type=int, default=9)
parser.add_argument("--sn", action='store_true', help="apply spectral norm")
parser.add_argument('--lip', type=float, default=1., help='lipschitz coeff in spectral norm')
parser.add_argument('--clip', type=float, default=0., help='weight clip')



opt = parser.parse_args()
print(opt)

SAVE_PATH = 'images'

SAVE_PATH += f'_s{opt.img_size}'
SAVE_PATH += f'_e{opt.n_epochs}'
SAVE_PATH += f'_DDep{opt.d_depth}'


if opt.sn:
    SAVE_PATH += '_sn'
if opt.lip != 1:
    SAVE_PATH += f'_lip{opt.lip}'
if opt.clip > 0:
    SAVE_PATH += f'_clip{opt.clip}'

os.makedirs(SAVE_PATH, exist_ok=True)
cuda = True if torch.cuda.is_available() else False

def distributed_train(opt, net, netD):
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device=torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        net.cuda()
        netD.cuda()
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[opt.local_rank], output_device=opt.local_rank)
        netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[opt.local_rank], output_device=opt.local_rank)

    
    return net, netD

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def add_sn(m, sn=True):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer, sn))    

    if sn and isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        return spectral_norm_flex(m, lip=opt.lip)
    else:
        return m

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator

'''
generator = Generator()
discriminator = MLPMixer(
    image_size = opt.img_size,
    channels = 3,
    patch_size = 8,
    dim = 256,
    depth = opt.d_depth,
    num_classes = 1
)
'''


generator = ...
discriminator = ...

discriminator = torch.nn.Sequential(
    discriminator,
    torch.nn.Sigmoid()
)

# Initialize weights

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if opt.sn:
    discriminator = add_sn(discriminator, sn=True)
print(discriminator)
 
# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)


len_dataset = ...

even = list(range(0, len_dataset, 2))
odd = list(range(1, len_dataset, 2))

dataset_mode1 = torch.utils.data.Subset(..., indices = even)
dataset_mode2 = torch.utils.data.Subset(..., indices = odd)
twomode_dataset = torch.utils.data.ConcatDataset([dataset_mode1, dataset_mode2])


dataloader_half = torch.utils.data.DataLoader(
    dataset_mode1,
    batch_size=opt.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers = 2
)

dataloader_full = torch.utils.data.DataLoader(
    twomode_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers = 2
)

# Optimizers
optimizer_G= torch.optim.AdamW(generator.parameters(), lr=1e-4, betas=(opt.b1, opt.b2), weight_decay=1e-4)
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=4e-4, betas=(opt.b1, opt.b2), weight_decay=1e-4)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH, 'tensorboard'))
# ----------
#  Training
# ----------
eps = 0
generator.train()
discriminator.train()
n_itr = 0
for epoch in range(opt.n_epochs):
    if epoch > opt.n_epochs / 2:
        dataloader = dataloader_full
    else:
        dataloader = dataloader_half

    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        g_loss = (torch.log(1 - discriminator(gen_imgs) + eps)).mean()  
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        with torch.no_grad():
            gen_imgs = generator(z)

        d_loss = ((torch.log(eps + discriminator(real_imgs))).mean() + (torch.log(1 + eps - discriminator(gen_imgs))).mean()) * -1  

        if torch.any(torch.isnan(d_loss)):
            exit()

        d_loss.backward()
        optimizer_D.step()

        writer.add_scalar('Loss/G', g_loss.item(), n_itr)
        writer.add_scalar('Loss/D', -d_loss.item(), n_itr)

        if opt.local_rank == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        if opt.clip > 0:
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip, opt.clip)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0 and opt.local_rank == 0:
            save_image(gen_imgs.data[:25], os.path.join(SAVE_PATH, "%d.png" % batches_done), nrow=5, normalize=True)

        n_itr += 1
    
    if (epoch+1) % 10 == 0:
        os.makedirs(os.path.join(SAVE_PATH, 'ckp'), exist_ok=True)
        torch.save(generator.state_dict(), os.path.join(SAVE_PATH, 'ckp', f'{epoch+1}_G.pth'))
        torch.save(discriminator.state_dict(), os.path.join(SAVE_PATH, 'ckp', f'{epoch+1}_D.pth'))

