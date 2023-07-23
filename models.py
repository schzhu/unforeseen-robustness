import math

import torch.nn as nn
from torchvision import models

MODEL_NAMES = ['resnet18small', 'resnet18', 'resnet50']


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


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.arch = args.arch
        self.backbone = self._load_backbone(args.arch, args.num_classes)

    def _load_backbone(self, arch, num_classes):
        if arch == 'resnet18small':
            # tailored resnet18 for small input sizes
            backbone = models.__dict__['resnet18'](num_classes=num_classes)
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()

        elif arch in ['resnet18', 'resnet50']:
            backbone = models.__dict__[arch](num_classes=num_classes)  # ,zero_init_residual=True

        elif arch == 'resnet18latent':
            backbone = models.__dict__['resnet18']()  # ,zero_init_residual=True
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            n_features = backbone.fc.weight.shape[1]
            backbone.fc = nn.Identity()
            self.head_cls = nn.Linear(n_features, num_classes, bias=True)
            self.num_backbone_features = n_features

        else:
            raise Exception(f'Not implemented model architecture {arch}')

        return backbone

    def forward(self, x, with_latent=False):
        if with_latent:
            assert self.arch == 'resnet18latent'
            z = self.backbone(x)
            out = self.head_cls(z)
            return out, z
        else:
            out = self.backbone(x)
            return out


def load_mlp(n_in, n_hidden, n_out, num_layers=3,
             inner_bn=True, last_bn=True, last_bias=None, last_sigmoid=False):
    if num_layers == 1:
        n_hidden = n_in
    if last_bias is None:
        last_bias = not last_bn
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        if inner_bn:
            layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=last_bias))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    mlp = nn.Sequential(*layers)
    reset_parameters(mlp)
    return mlp
