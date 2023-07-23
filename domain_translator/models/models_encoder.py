"""
https://github.com/w86763777/pytorch-gan-collections/blob/master/source/models/wgangp.py
"""
import torch
import torch.nn as nn
from torch.nn import init
import math


class TransformEncoderDefault(nn.Module):
    """ Encoding a pair of transformed images into latent variables (params)
    input  : two image tensors (B x 3 x H x W)
    output : B x latent_dim
    """
    def __init__(self):
        super(TransformEncoderDefault, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # downsampling
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.projector = load_mlp(8192, 512, 512, num_layers=2, last_bn=True)
        self.predictor = load_mlp(512, 128, 512, num_layers=2, last_bn=False)
        init_weights_of_model(self.model)
        init_weights_of_model(self.projector)
        init_weights_of_model(self.predictor)

    def forward(self, img1, img2, return_latent=False):
        x_ = self.model(torch.cat((img1, img2)))
        x1, x2 = torch.tensor_split(x_, 2)
        x = torch.cat((x1, x2), dim=1)
        out = self.projector(x.flatten(1))
        if return_latent:
            return x1, x2, out
        else:
            return out


class TransformEncoderDefault64(nn.Module):
    """ Encoding a pair of transformed images into latent variables (params)
    input  : two image tensors (B x 3 x H x W)
    output : B x latent_dim
    """
    def __init__(self):
        super(TransformEncoderDefault64, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.projector = load_mlp(16384, 512, 512, num_layers=2, last_bn=True)
        self.predictor = load_mlp(512, 128, 512, num_layers=2, last_bn=False)
        init_weights_of_model(self.model)
        init_weights_of_model(self.projector)
        init_weights_of_model(self.predictor)

    def forward(self, img1, img2, return_latent=False):
        x_ = self.model(torch.cat((img1, img2)))  # .sum(dim=[2, 3])
        x1, x2 = torch.tensor_split(x_, 2)
        x = torch.cat((x1, x2), dim=1)
        out = self.projector(x.flatten(1))
        if return_latent:
            return x1, x2, out
        else:
            return out


class TransformEncoderDefault96(nn.Module):
    """ Encoding a pair of transformed images into latent variables (params)
    input  : two image tensors (B x 3 x H x W)
    output : B x latent_dim
    """
    def __init__(self):
        super(TransformEncoderDefault96, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.projector = load_mlp(9216, 512, 512, num_layers=2, last_bn=True)
        self.predictor = load_mlp(512, 128, 512, num_layers=2, last_bn=False)
        init_weights_of_model(self.model)
        init_weights_of_model(self.projector)
        init_weights_of_model(self.predictor)

    def forward(self, img1, img2, return_latent=False):
        x_ = self.model(torch.cat((img1, img2)))
        x1, x2 = torch.tensor_split(x_, 2)
        x = torch.cat((x1, x2), dim=1)
        out = self.projector(x.flatten(1))
        if return_latent:
            return x1, x2, out
        else:
            return out


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


def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()

        if isinstance(m, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)


def load_mlp(n_in, n_hidden, n_out, num_layers=3,
             inner_bn=True, last_bn=True, last_bias=None, last_sigmoid=False):
    if num_layers == 1:
        n_hidden = n_in
    if last_bias is None:
        last_bias = not last_bn
    layers = []
    for i in range(num_layers-1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        if inner_bn:
            layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=last_bias))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    mlp = nn.Sequential(*layers)
    # reset_parameters(mlp)
    return mlp


Encoders = {
    'default': TransformEncoderDefault(),
    'default64': TransformEncoderDefault64(),
    'default96': TransformEncoderDefault96(),
}
