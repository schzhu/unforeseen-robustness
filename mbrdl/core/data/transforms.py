import math
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

import random
import torch
import torch.nn as nn
import torch.nn.functional as NF
from torch import Tensor

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import kornia
import kornia.augmentation as K
# import kornia.augmentation.functional as KF
import kornia.geometry.transform as KGT
import kornia.enhance as KE

from kornia.augmentation.utils import _transform_input, _validate_input_dtype
from kornia.enhance import adjust_hue, adjust_saturation
from typing import Dict
import math

# mean and std of each dataset
STATS = {
    'cifar10': (torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2023, 0.1994, 0.2010])),
    'cifar10a': (torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2023, 0.1994, 0.2010])),
    'cifar10b': (torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2023, 0.1994, 0.2010])),
    'cifar100': (torch.tensor([0.5071, 0.4867, 0.4408]), torch.tensor([0.2675, 0.2565, 0.2761])),
    'mnist': (torch.tensor([0.1307, 0.1307, 0.1307]), torch.tensor([0.3081, 0.3081, 0.3081])),
    'svhn': (torch.tensor([0.4376, 0.4437, 0.4728]), torch.tensor([0.1980, 0.2010, 0.1970])),
    'stl10': (torch.tensor([0.43, 0.42, 0.39]), torch.tensor([0.27, 0.26, 0.27])),
    # 'celeba': (torch.tensor([0., 0., 0.]), torch.tensor([1., 1., 1.])),
    'tinyimagenet': (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
    'objectron': (torch.tensor([0.577, 0.512, 0.467]), torch.tensor([0.250, 0.243, 0.243])),
    'imagenet': (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
}


class MultiView:
    def __init__(self, transform, num_views=2):
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


class Transformer:
    def __init__(self, transform_name, dataset_tar=None, dataset_src=None):  #, device):
        """
        :param transform_name:
            1. single transform: rotate, color, crop, flip, etc.
            2. sub-policies of AutoAugment: sp1, sp2, ...
            3. RandAugment or AutoAugment: randaugment, autoaugment
        """
        self.transform_name = transform_name
        # self.convert_to_uint8 = False
        self.transform = self._load_transform(transform_name, dataset_tar)

        # image normalizer
        if dataset_tar is not None:
            self.normalize_tar = T.Normalize(*STATS[dataset_tar])
            self.denormalize_tar = K.Denormalize(*STATS[dataset_tar])
        if dataset_src is not None:
            self.normalize_src = T.Normalize(*STATS[dataset_src])
            self.denormalize_src = K.Denormalize(*STATS[dataset_src])

        # scale the tanh due to non-standard (0.5) normalization
        # self.scale_tar = Transformer.get_scale(dataset_tar)

    def _single_transforms(self):
        return {
            'rotate': K.RandomRotation(30., p=1., resample='NEAREST')
        }

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]):
        return {
            # 'Rotate':
        }

    def _load_transform(self, transform_name, dataset_tar):
        single_transforms = self._single_transforms()

        if transform_name in single_transforms.keys():
            transform = single_transforms[transform_name]

        elif transform_name.startswith('sp'):
            raise NotImplemented()

        elif transform_name == 'randaugment':
            # self.convert_to_uint8 = True
            transform = RandAugmentBatch()

        else:
            raise Exception(f'Unknown transform {transform_name}')
        return transform

    # @staticmethod
    # def get_scale(dataset):
    #     lower, upper = 0., 0.
    #     means, stds = STATS[dataset]
    #     for mean, std in zip(means, stds):
    #         lower = min(lower, (0 - mean) / std)
    #         upper = max(upper, (1 - mean) / std)
    #     scale = max(lower.abs(), upper.abs())
    #     return scale

    def apply(self, imgs1, imgs2=None, denorm_1=None, denorm_2=None):
        """
        refers to https://github.com/pytorch/vision/issues/9
        :param img1: float32, unnormalized
        :param img2: float32, unnormalized
        :return: float32
        """
        # if self.convert_to_uint8:
        #     imgs1 = (imgs1 * 255).to(torch.uint8)
        #     imgs2 = (imgs2 * 255).to(torch.uint8)
        # if self.convert_to_uint8:
        #     out1 = out1.float() / 255
        #     out2 = out2.float() / 255
        if imgs2 is None:
            if denorm_1 == 'src':
                imgs1 = self.denormalize_src(imgs1)
            elif denorm_1 == 'tar':
                imgs1 = self.denormalize_tar(imgs1)
            elif denorm_1 is None:
                imgs1 = imgs1
            else:
                raise Exception(f'Unknown denormalization argument')

            out1 = self.transform(imgs1)

            if denorm_1 == 'src':
                out1 = self.normalize_src(out1)
            elif denorm_1 == 'tar':
                out1 = self.normalize_tar(out1)
            elif denorm_1 is None:
                out1 = out1
            else:
                raise Exception(f'Unknown denormalization argument')

            return out1

        else:
            if denorm_1 == 'src':
                imgs1 = self.denormalize_src(imgs1)
            elif denorm_1 == 'tar':
                imgs1 = self.denormalize_tar(imgs1)
            elif denorm_1 is None:
                imgs1 = imgs1
            else:
                raise Exception(f'Unknown denormalization argument')

            if denorm_2 == 'src':
                imgs2 = self.denormalize_src(imgs2)
            elif denorm_2 == 'tar':
                imgs2 = self.denormalize_tar(imgs2)
            elif denorm_2 is None:
                imgs2 = imgs2
            else:
                raise Exception(f'Unknown denormalization argument')

            state = torch.get_rng_state()
            out1 = self.transform(imgs1)
            torch.set_rng_state(state)
            out2 = self.transform(imgs2)

            if denorm_1 == 'src':
                out1 = self.normalize_src(out1)
            elif denorm_1 == 'tar':
                out1 = self.normalize_tar(out1)
            elif denorm_1 is None:
                out1 = out1
            else:
                raise Exception(f'Unknown denormalization argument')

            if denorm_2 == 'src':
                out2 = self.normalize_src(out2)
            elif denorm_2 == 'tar':
                out2 = self.normalize_tar(out2)
            elif denorm_2 is None:
                out2 = out2
            else:
                raise Exception(f'Unknown denormalization argument')

            return out1, out2


def _apply_op(
    imgs: Tensor, op_name: str, magnitudes: Tensor, mode: str, padding_mode: str):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        magnitudes = magnitudes[:, None]
        params = torch.atan(magnitudes)
        params = torch.cat((params, torch.zeros_like(params)), dim=1)
        imgs = KGT.shear(
            imgs,
            shear=params,
            mode=mode,
            padding_mode=padding_mode,
        )
    elif op_name == "ShearY":
        # magnitudes should be arctan(magnitudes)
        # See above
        magnitudes = magnitudes[:, None]
        params = torch.atan(magnitudes)
        params = torch.cat((torch.zeros_like(params), params), dim=1)
        imgs = KGT.shear(
            imgs,
            shear=params,
            mode=mode,
            padding_mode=padding_mode,
        )
    elif op_name == "TranslateX":
        magnitudes = magnitudes[:, None]
        params = magnitudes  # .int()
        params = torch.cat((params, torch.zeros_like(params)), dim=1)
        imgs = KGT.translate(
            imgs,
            translation=params,
            mode=mode,
            padding_mode=padding_mode
        )
    elif op_name == "TranslateY":
        magnitudes = magnitudes[:, None]
        params = magnitudes  # .int()
        params = torch.cat((torch.zeros_like(params), params), dim=1)
        imgs = KGT.translate(
            imgs,
            translation=params,
            mode=mode,
            padding_mode=padding_mode
        )
    elif op_name == "Rotate":
        imgs = KGT.rotate(imgs, magnitudes, mode=mode, padding_mode=padding_mode)
    elif op_name == "Brightness":
        magnitudes = magnitudes[:, None, None, None]
        imgs = imgs * (1.0 + magnitudes) + torch.zeros_like(imgs) * (- magnitudes)
        imgs = imgs.clamp(0, 1.)
    elif op_name == "Color":
        imgs = KE.adjust_saturation(imgs, 1.0 + magnitudes)
    elif op_name == "Contrast":
        imgs = KE.adjust_contrast(imgs, 1.0 + magnitudes)
    elif op_name == "Sharpness":
        imgs = KE.sharpness(imgs, 1.0 + magnitudes)
    elif op_name == "Posterize":
        imgs = KE.posterize(imgs, magnitudes.int())
    elif op_name == "Solarize":
        magnitudes = magnitudes / 255.  # original implementation uses int8 inputs
        imgs = KE.solarize(imgs, magnitudes)
    elif op_name == "AutoContrast":
        imgs = F.autocontrast(imgs)
    elif op_name == "Equalize":
        imgs = KE.equalize(imgs)
    elif op_name == "Invert":
        imgs = KE.invert(imgs)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return imgs


class RandAugmentBatch:
    """
        A modified version based on https://pytorch.org/vision/main/_modules/torchvision/transforms/autoaugment.html#AutoAugmentPolicy
        Made for batch operation with same_on_batch=False
    """
    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def get_params(self, op_meta, batch_size):
        """Get mask and params

        :return:
            mask: Binary matrix, K x B
            params: List of transform param vectors, K x (B,),
                    each representing the signed magnitude
        """
        num_all_ops = len(op_meta)
        mask = torch.zeros((num_all_ops, batch_size), dtype=torch.bool)
        params = [[] for _ in range(num_all_ops)]
        for i in range(batch_size):
            # ops_indices = np.random.choice(np.arange(0, num_all_ops),
            #                                self.num_ops,
            #                                replace=False)  # cannot use np b/c randomness not controlled
            perm = torch.randperm(num_all_ops)
            ops_indices = perm[:self.num_ops]

            mask[ops_indices, i] = True
            for op_index in ops_indices:
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                params[op_index].append(magnitude)

        return mask, params


    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str,
                            Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def __call__(self, imgs):
        """
        :param imgs: Tensor (B x C x H x W), unnormalized
        :return:
        """
        fill = self.fill
        batch_size, channels, height, width = imgs.shape
        if isinstance(imgs, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        # generate mask and params
        mask, params = self.get_params(op_meta=op_meta, batch_size=batch_size)

        # use mask and params to apply op sequentially
        imgs = imgs.clamp(0, 1.)  # otherwise netG's output goes as low as 2.98e-8
        for op_index in range(len(op_meta)):
            imgs_to_apply = imgs[mask[op_index]]
            if len(imgs_to_apply) == 0:
                continue
            op_name = list(op_meta.keys())[op_index]
            magnitudes = torch.tensor(params[op_index]).to(imgs.device)
            imgs[mask[op_index]] = _apply_op(imgs_to_apply,
                                             op_name,
                                             magnitudes,
                                             mode='bilinear',
                                             padding_mode='zeros')
        return imgs
