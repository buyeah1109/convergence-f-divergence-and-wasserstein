import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNet(torch.nn.Module):
    def __init__(self, channel, img_size, num_class):
        super(ConvNet, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channel, 64, bn=False),
            *discriminator_block(64, 128, bn=False),
            *discriminator_block(128, 256, bn=False),
            *discriminator_block(256, 512, bn=False),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, num_class), nn.Sigmoid())

    def forward(self, img, forward_feature=False):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        if forward_feature:
            return out
        validity = self.adv_layer(out)

        return validity
    
class SoftBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SoftBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SoftBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(SoftBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SoftResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SoftResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, forward_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if forward_feature:
            return out
        out = self.linear(out)
        return out

def soft_cifar_ResNet18(num_classes=10):
    return SoftResNet(SoftBasicBlock, [2,2,2,2], num_classes=num_classes)

class DCGenerator(nn.Module):
    def __init__(self, img_size, latent_dim, channels=3):
        super(DCGenerator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class MLPGenerator2(nn.Module):
    def __init__(self):
        super(MLPGenerator2, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(256, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(28*28*1))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], 28*28)
        return img
    
class MLP(nn.Module):
    def __init__(self, img_size, channel, num_class):
        super(MLP, self).__init__()
        self.img_shape = (channel, img_size, img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_class),
        )

    def forward(self, img, forward_feature=False):
        img_flat = img.view(img.size(0), -1)
        if forward_feature:
            return self.model[:-1](img_flat)
        validity = self.model(img_flat)

        return validity


class ResBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResBlock, self).__init__()

        self.conv_block = self.build_conv_block(in_dim)

    def build_conv_block(self, in_dim):
        conv_block = []
        conv_block +=[                                                                    # 1 full Resnet Block
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0, bias=True),
                        nn.BatchNorm2d(in_dim),
                        nn.ReLU(True),
                        #nn.Dropout(0.5),                                               #dont use dropout- Niranjan
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0, bias=True),
                        nn.BatchNorm2d(in_dim)
                    ]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResGenerator(nn.Module):
    def __init__(self, opt, depth):
        super(ResGenerator, self).__init__()

        self.init_size = opt.img_size // (2 ** 2)
        self.in_channel = 128
        self.l1 = nn.Linear(opt.latent_dim, self.in_channel * self.init_size ** 2)
        self.l_norm = nn.BatchNorm2d(self.in_channel)
        # self.conv_l = nn.Conv2d(int(self.in_channel / (2 ** depth)), 3, 1)

        self.conv_blocks = []
        for i in range(depth):
            self.conv_blocks += [ResBlock(int(self.in_channel))]

        # self.conv_blocks += [nn.Upsample(scale_factor=2),
        #                     ResBlock(int(self.in_channel / mul), int(self.in_channel / (2 * mul)))]
        self.conv_blocks +=[   
                nn.ConvTranspose2d(self.in_channel, self.in_channel // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),  #upsampling layer-1
                nn.BatchNorm2d(self.in_channel // 2),                                                                                #
                nn.ReLU(True),                                                                                  #

                nn.ConvTranspose2d(self.in_channel // 2, self.in_channel // 4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),   #upsampling layer-2
                nn.BatchNorm2d(self.in_channel // 4),                                                                                 #
                nn.ReLU(True),                                                                                  #

                nn.ReflectionPad2d(3),                                              #Reflection padding applied

                nn.Conv2d(self.in_channel // 4, 3, kernel_size=7, padding=0),                         #7X7 conv applied; 3 filters/outputs

                nn.Tanh()                                                           #Tanh activation function used finally

            ]
        
        self.conv_blocks = nn.Sequential(
            *self.conv_blocks,
            # self.conv_l,
            # nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.in_channel, self.init_size, self.init_size)
        out = self.l_norm(out)

        img = self.conv_blocks(out)
        return img

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.dim = 512
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.dim, self.init_size, self.init_size)

        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.LayerNorm(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 64, bn=False),
            *discriminator_block(64, 128, bn=False),
            *discriminator_block(128, 256, bn=False),
            *discriminator_block(256, 512, bn=False),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class MLPGenerator(nn.Module):
    def __init__(self, opt):
        super(MLPGenerator, self).__init__()
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class MLPDiscriminator(nn.Module):
    def __init__(self, opt):
        super(MLPDiscriminator, self).__init__()
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

