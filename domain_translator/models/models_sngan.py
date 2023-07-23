"""
https://github.com/w86763777/pytorch-gan-collections/blob/master/source/models/sngan.py
"""
import torch
import torch.nn as nn
from torch.nn import init
import math
from torch.nn.utils.spectral_norm import spectral_norm

def init_weights_of_model(m, init_type='normal', init_gain=0.02, sn=False):
    classname = m.__class__.__name__

    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        # The default init type is 'normal', which is used for both CycleGAN paper and pix2pix paper.
        # However, in some cases, xavier and kaiming might work better for some applications.
        # Thus, try all of them for experiment.
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        if sn:
            spectral_norm(m)

    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ Generator
    input  : N x channels x height x width (image)
    output : N x channels x height x width (image)
    """
    def __init__(self, input_channel=3, num_residual_blocks=2, num_features=64):
        super(Generator, self).__init__()

        # self.output_scale = 1.
        # self.output_scale = output_scale
        # channels = input_channel

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        model += [
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(256)]

        # Upsampling
        model += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]

        # Output layer
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(128, 3, kernel_size=3),
                  nn.Sigmoid()]  # nn.Tanh() -> nn.Sigmoid(), since we normalize later

        self.model = nn.Sequential(*model)
        init_weights_of_model(self.model, 'normal')

    def forward(self, x):
        return self.model(x)  # * self.output_scale


def discriminator_block(in_filters, out_filters, normalize=True):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_filters, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator32(nn.Module):
    """ Discriminator
    input  : N x channels x height x width (image)
    output : N x 1 x 1 x 1 (probability)
    """
    def __init__(self, input_channel=3, num_features=64):
        super(Discriminator32, self).__init__()

        # Calculate output shape of image discriminator (PatchGAN)
        # self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.model = nn.Sequential(
            *discriminator_block(3, num_features, normalize=False),
            *discriminator_block(num_features, num_features*2),
            *discriminator_block(num_features*2, num_features*4),
            # *discriminator_block(num_features * 4, num_features * 8),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(num_features*4, 1, 4, stride=2, padding=0)
        )
        init_weights_of_model(self.model, 'normal', sn=True)

    def forward(self, img):
        return self.model(img)


class Discriminator64(nn.Module):
    """ Discriminator
    input  : N x channels x height x width (image)
    output : N x 1 x 1 x 1 (probability)
    """
    def __init__(self, input_channel, num_features=64):
        super(Discriminator64, self).__init__()

        # Calculate output shape of image discriminator (PatchGAN)
        # self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.model = nn.Sequential(
            *discriminator_block(3, num_features, normalize=False),
            *discriminator_block(num_features, num_features*2),
            *discriminator_block(num_features*2, num_features*4),
            *discriminator_block(num_features * 4, num_features * 8),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(num_features*8, 1, 4, stride=2, padding=0)
        )
        # init_weights_of_model(self.model, 'xavier')
        init_weights_of_model(self.model, 'normal', sn=True)

    def forward(self, img):
        return self.model(img)




class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2))
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = nn.Linear(128, 1, bias=False)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        spectral_norm(self.linear)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


class ResDiscriminator64(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128, down=True),
            nn.ReLU())
        self.linear = nn.Linear(128, 1, bias=False)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)
        spectral_norm(self.linear)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


class Discriminator_ref(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = nn.Linear(M // 8 * M // 8 * 512, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x, *args, **kwargs):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

class Discriminator32_ref(Discriminator_ref):
    def __init__(self):
        super().__init__(M=32)

