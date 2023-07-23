"""
https://github.com/tanimutomo/cifar10-c-eval
"""
import os
import torch
import torch.nn as nn
import models
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import loaders
from utils import AverageMeter, accuracy
import numpy as np
import pprint


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10-c')
    parser.add_argument('--data_dir', type=str, default='work_dirs/datasets')
    parser.add_argument('--log_dir', type=str, default='work_dirs/logs')
    parser.add_argument('--arch', type=str, default='resnet18small', choices=models.MODEL_NAMES,
                        help="classifier architecture (default: resnet18small)")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_dir', type=str)

    args = parser.parse_args()
    return args


def load_state(ckpt, model):
    for k, v in model.items():
        if isinstance(v, nn.parallel.DistributedDataParallel):
            v = v.module

        if hasattr(v, 'state_dict'):
            v.load_state_dict(ckpt[k])

        if type(v) is list and hasattr(v[0], 'state_dict'):
            for i, x in enumerate(v):
                x.load_state_dict(ckpt[k][i])

        if type(v) is dict and k == 'ss_predictor':
            for y, x in v.items():
                x.load_state_dict(ckpt[k][y])
    return model


def eval(model, loader, args):
    model.eval()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for itr, (x, y) in enumerate(loader):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, dtype=torch.int64, non_blocking=True)

            z = model(x)
            # loss = F.cross_entropy(z, y)
            acc, _ = accuracy(z, y, topk=(1, 5))
            acc_meter.update(acc.item())
    return acc_meter.avg


def main(args):
    # load model
    model = models.Classifier(args)
    model_dir = os.path.join(args.log_dir, args.model_dir)
    ckpt = torch.load(model_dir, map_location='cpu')
    model.load_state_dict(ckpt['backbone'])  # new model
    # model = load_state(ckpt, model)
    model = model.to(args.device)
    model.eval()

    # load data and evaluate
    if args.dataset == 'cifar10-c':
        print('Eval CIFAR10-C:')
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
        accs = dict()

        for ci, cname in enumerate(corruptions):
            # load dataset
            dataset = loaders.load_ood_dataset(dataset='cifar10-c', datadir=args.data_dir,
                                               name=cname)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
            acc = eval(model, loader, args)
            accs[f'{cname}'] = acc

            print(f'{cname}: {acc:.2f}')

        avg = np.mean(list(accs.values()))
        accs['avg'] = avg
        print(f'avg: {avg:.2f}')

        pprint.pprint(accs)


if __name__ == "__main__":
    args = arg_parse()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
