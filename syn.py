import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.nn.utils.parametrizations import spectral_norm
import matplotlib

wd = False
sn = False
d_sigmoid = False

f_div = True
pearson = False
reverse_kl = False
hellin = True
kl = False
SAVE_PATH = ...
os.makedirs(SAVE_PATH, exist_ok= True)

def generate_batch(batchlen, twomode=False):

    if twomode:
      samples = torch.rand(size=(batchlen-1, 1))
      means = torch.zeros((1, 2))
      for sample in samples:
        if sample <= 0.2:
          means = torch.vstack([means, torch.FloatTensor([0, 0])])
        elif sample <= 0.4:
          means = torch.vstack([means, torch.FloatTensor([0, 1])])
        elif sample <= 0.6:
          means = torch.vstack([means, torch.FloatTensor([0, -1])])
        elif sample <= 0.8:
          means = torch.vstack([means, torch.FloatTensor([1, 0])])
        else:
          means = torch.vstack([means, torch.FloatTensor([-1, 0])])

    else:
      samples = torch.multinomial(torch.tensor([0.,0.,1.,0.,0.]), 2 * batchlen, replacement=True)
      means =  (samples - 2).view(batchlen,2).type(torch.FloatTensor)

    return torch.normal(means, 0.05)

class Generator(nn.Module):
    def __init__(self, WDTH=0, DPTH=0, PRIOR_N=1, PRIOR_STD=1.):
        super().__init__()
        self.PRIOR_N = PRIOR_N
        self.PRIOR_STD = PRIOR_STD
        # First layer
        self.fc1 = nn.Linear(PRIOR_N, WDTH)
        # Hidden layers
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        # Transform list into layers using nn.ModuleList
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        # Output layer
        self.fc2 = nn.Linear(WDTH, 2)

    def __call__(self, z):
        h = F.leaky_relu(self.fc1(z), 0.2)
        for hidden_layer in self.hidden_layers:
            h = F.leaky_relu(hidden_layer(h), 0.2)

        # return self.fc2(h)
        return torch.tanh(self.fc2(h)) * 1.5

    def generate(self, batchlen):
        z = torch.normal(torch.zeros(batchlen, self.PRIOR_N), self.PRIOR_STD)
        return self.__call__(z)

class Discriminator(nn.Module):
    def __init__(self, WDTH=0, DPTH=0):
        super().__init__()
        # First layer
        self.fc1 = nn.Linear(2, WDTH)
        # Hidden layers
        self.hidden_layers = []
        for _ in range(DPTH):
            self.hidden_layers.append(nn.Linear(WDTH, WDTH))
        # Transform list into layers using nn.ModuleList
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        # Output layer
        self.fc2 = nn.Linear(WDTH, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        for hidden_layer in self.hidden_layers:
            h = F.leaky_relu(hidden_layer(h), 0.2)
        if reverse_kl:
            return -torch.exp(-1 * self.fc2(h))
        elif d_sigmoid:
            return torch.sigmoid(self.fc2(h))
        elif hellin:
            return 1 - torch.exp(-self.fc2(h))
        else:
            return self.fc2(h)


def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))

    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        return spectral_norm(m)
    elif isinstance(m, torch.nn.LayerNorm):
        return torch.nn.Identity()
    else:
        return m

def pearson_conjugate(value):
    return 0.25 * value ** 2 + value

def reverse_kl_conjugate(value):
    return -1 - torch.log(-1 * value)

def square_hellinger_conjugate(v):
    return v / (1 - v)

def kl_conjugate(v):
    return torch.exp(v - 1)

def GAN(TRAIN_RATIO=1, N_ITER=40001, BATCHLEN=128,
        WDTH_G=0, DPTH_G=0, WDTH_D=0, DPTH_D=0,
        PRIOR_N=1, PRIOR_STD=1., frame=1000):
    """
    TRAIN_RATIO : int, number of times to train the discriminator between two generator steps
    N_ITER : int, total number of training iterations for the generator
    BATCHLEN : int, Batch size to use
    WDTH_G : int, width of the generator (number of neurons in hidden layers)
    DPTH_G : int, number of hidden FC layers of the generator
    WDTH_D : int, width of the discriminator (number of neurons in hidden layers)
    DPTH_D : int, number of hidden FC layers of the discriminator
    PRIOR_N : int, dimension of input noise
    PRIOR_STD : float, standard deviation of p(z)
    frame : int, display data each 'frame' iteration
    """

    G = Generator(WDTH=WDTH_G, DPTH=DPTH_G, PRIOR_N=PRIOR_N, PRIOR_STD=PRIOR_STD)
    # G = RandomLinearGenerator(PRIOR_N=PRIOR_N, PRIOR_STD=PRIOR_STD,
    #                           mu1=torch.FloatTensor([1, 1]),
    #                           mu2=torch.FloatTensor([0, 0]))
    solver_G = torch.optim.SGD(G.parameters(), lr=1e-3, momentum=0.9)
    D = Discriminator(WDTH=WDTH_D, DPTH=DPTH_D)
    if sn:
        D = add_sn(D)
    solver_D = torch.optim.SGD(D.parameters(), lr=1e-3, momentum=0.9)
    second_ratio = 2 # smaller than N_ITER
    # G, D = G.cuda(), D.cuda()
    eps = 1e-7
    for i in tqdm(range(N_ITER)):
        # train the discriminator
        for _ in range(TRAIN_RATIO):
            D.zero_grad()
            real_batch = generate_batch(BATCHLEN, twomode = i>N_ITER / second_ratio)

            fake_batch = G.generate(BATCHLEN)

            # Compute here the discriminator loss, using functions like torch.sum, torch.exp, torch.log,
            # torch.softplus, using real_batch and fake_batch
            h_real = D(real_batch)
            h_fake = D(fake_batch)
            if f_div:
                if reverse_kl:
                    loss_fake = torch.mean(reverse_kl_conjugate(h_fake))
                elif pearson:
                    loss_fake = torch.mean(pearson_conjugate(h_fake))
                elif hellin:
                    loss_fake = torch.mean(square_hellinger_conjugate(h_fake))
                elif kl:
                    loss_fake = torch.mean(kl_conjugate(h_fake))

                disc_loss = (torch.mean(h_real) - loss_fake) * -1
            elif wd:
                loss_real = torch.mean(h_real)
                loss_fake = torch.mean(h_fake)
                disc_loss = (loss_real - loss_fake) * -1
            else:
                loss_real = torch.mean(torch.log(h_real + eps))
                loss_fake = torch.mean(torch.log(1 - h_fake + eps))
                disc_loss = (loss_real + loss_fake) * -1

            disc_loss.backward()
            solver_D.step()
            # if wd:
            #     for p in D.parameters():
            #         p.data.clamp_(-0.01, 0.01)
        # train the generator
        G.zero_grad()
        fake_batch = G.generate(BATCHLEN)
        # Compute here the generator loss, using fake_batch
        h_fake = D(fake_batch)
        if wd:
            gen_loss = torch.mean(h_fake) * -1
        elif f_div:
            if reverse_kl:
                gen_loss = -1 * torch.mean(reverse_kl_conjugate(h_fake))
            elif pearson:
                gen_loss = -1 * torch.mean(pearson_conjugate(h_fake))
            elif hellin:
                gen_loss = -1 * torch.mean(square_hellinger_conjugate(h_fake))
            elif kl:
                gen_loss = -1 * torch.mean(kl_conjugate(h_fake))


        else:
            gen_loss = torch.mean(torch.log(1 - h_fake + eps))

        gen_loss.backward()
        solver_G.step()
        if i%frame == 0:
            plt.figure()
            print('step {}: discriminator: {:.3e}, generator: {:.3e}'.format(i, float(disc_loss), float(gen_loss)))
            # plot the result
            real_batch = generate_batch(1024, twomode = True)
            fake_batch = G.generate(1024).detach()
            plt.scatter(real_batch[:,0].cpu(), real_batch[:,1].cpu(), s=4.0, label='Real data')
            plt.scatter(fake_batch[:,0].cpu(), fake_batch[:,1].cpu(), s=40.0, marker='x', label='Generated data')

            # plt.legend(loc='lower right')
            plt.show()
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_PATH ,f'{i}.png'))
            plt.close()