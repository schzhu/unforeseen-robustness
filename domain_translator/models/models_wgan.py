"""
https://github.com/w86763777/pytorch-gan-collections/blob/master/source/models/wgangp.py
"""
import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F


def create_generator(image_size, num_residual_blocks=2):
    if image_size == 32:
        return Generator32(num_residual_blocks)
    elif image_size == 64:
        return Generator64(num_residual_blocks)
    elif image_size == 96:
        return Generator64(num_residual_blocks)
    else:
        raise NotImplementedError(f"Unsupported image size: {image_size}")


def create_discriminator(image_size):
    if image_size == 32:
        return Discriminator32()
    elif image_size == 64:
        return Discriminator32()
    elif image_size == 96:
        return Discriminator32()
    else:
        raise NotImplementedError(f"Unsupported image size: {image_size}")


def init_weights_of_model(m, init_type='normal', init_gain=0.02):
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

    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Generator32(nn.Module):
    """ Generator
    input  : N x channels x height x width (image)
    output : N x channels x height x width (image)
    """
    def __init__(self, num_residual_blocks=2):
        super(Generator32, self).__init__()
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
        init_weights_of_model(self.model)
        # reset_parameters(self.model)

    def forward(self, x):
        return self.model(x)  # * self.output_scale


class Generator64(nn.Module):
    """ Generator
    input  : N x channels x height x width (image)
    output : N x channels x height x width (image)
    """
    def __init__(self, num_residual_blocks=3):
        super(Generator64, self).__init__()
        # Initial convolution block
        model = [
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        model += [
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
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
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Output layer
        model += [
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Sigmoid()
        ]  # nn.Tanh() -> nn.Sigmoid(), since we normalize later

        self.model = nn.Sequential(*model)
        weights_init_normal(self.model)
        # reset_parameters(self.model)

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
    def __init__(self, num_features=64):
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
        init_weights_of_model(self.model)
        # reset_parameters(self.model)

    def forward(self, img):
        return self.model(img)


class Discriminator64(nn.Module):
    """ Discriminator
    input  : N x channels x height x width (image)
    output : N x 1 x 1 x 1 (probability)
    """
    def __init__(self):
        super(Discriminator64, self).__init__()

        # Calculate output shape of image discriminator (PatchGAN)
        # self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )
        weights_init_normal(self.model)

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
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

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
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

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
        self.linear = nn.Linear(128, 1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x


class ResidualBlockEncoder(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlockEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


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

    def forward(self, x, *args, **kwargs):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Discriminator32_ref(Discriminator_ref):
    def __init__(self):
        super().__init__(M=32)
