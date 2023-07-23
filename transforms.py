from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import kornia.augmentation as K
import kornia.geometry.transform as KGT
import kornia.enhance as KE

from loaders import STATS


def identity_map(x):
    return x


class Transformer:
    def __init__(self, transform_name, dataset_tar=None, dataset_src=None, device=None):  #
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
        self.normalize_tar = K.Normalize(*STATS[dataset_tar])
        self.denormalize_tar = K.Denormalize(*STATS[dataset_tar])
        if dataset_src is not None:
            self.normalize_src = K.Normalize(*STATS[dataset_src])
            self.denormalize_src = K.Denormalize(*STATS[dataset_src])

        self.device = device

        # scale the tanh due to non-standard (0.5) normalization
        # self.scale_tar = Transformer.get_scale(dataset_tar)

    def _single_transforms(self):
        return {
            'identity': T.Lambda(lambda x: x),
            'rotate': K.RandomRotation(15., p=1.),
            'perspective': K.RandomPerspective(p=1.),
            'affine': K.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2),
                                     shear=(-10., 10.), p=1.),
            'crop': K.RandomCrop((32, 32), padding=4, p=1.),
            'elastic_transform': K.RandomElasticTransform(p=1.),
            'fisheye': K.RandomFisheye(center_x=torch.tensor([-.3, .3]),
                                       center_y=torch.tensor([-.3, .3]),
                                       gamma=torch.tensor([.9, 1.]),
                                       p=1.),
            'thin_plate_spline': K.RandomThinPlateSpline(p=1.),
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
        elif transform_name == '3d':
            transform = T.Lambda(identity_map)
        elif transform_name == 'illumination':
            transform = T.Lambda(identity_map)
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

    def apply(self, imgs1, imgs2=None, denorm_1=None, denorm_2=None, idx=0):
        """
        refers to https://github.com/pytorch/vision/issues/9
        original input leaves intact, gradient is detached
        :param img1: float32, unnormalized
        :param img2: float32, unnormalized
        :return: float32
        """
        if imgs2 is None:
            if isinstance(imgs1, dict):  # 3d or illumination
                if isinstance(imgs1['extra_views'], list):
                    return imgs1['extra_views'][idx].to(self.device)
                elif isinstance(imgs1['extra_views'], torch.Tensor):
                    return imgs1['extra_views'][:, idx, :].to(self.device)

            imgs1 = imgs1.clone()
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
            imgs2 = imgs2.clone()
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

    def apply_seq(self, imgs_seq, denorm_seq):
        """
        refers to https://github.com/pytorch/vision/issues/9
        :param img1: float32, unnormalized
        :param img2: float32, unnormalized
        :return: float32
        """
        outs = []
        num = len(imgs_seq)
        for i in range(num):
            imgs, denorm = imgs_seq[i], denorm_seq[i]
            imgs = imgs.clone()
            if denorm == 'src':
                imgs = self.denormalize_src(imgs)
            elif denorm == 'tar':
                imgs = self.denormalize_tar(imgs)
            elif denorm is None:
                imgs = imgs
            else:
                raise Exception(f'Unknown denormalization argument')

            state = torch.get_rng_state()
            out = self.transform(imgs)
            if i < num - 1:
                torch.set_rng_state(state)

            if denorm == 'src':
                out = self.normalize_src(out)
            elif denorm == 'tar':
                out = self.normalize_tar(out)
            elif denorm is None:
                out = out
            else:
                raise Exception(f'Unknown denormalization argument')

            outs.append(out)

        return outs


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


class RandAugmentBatch(torch.nn.Module):
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

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> \
            Dict[str, Tuple[Tensor, bool]]:
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

    def forward(self, imgs):
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
