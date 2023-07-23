"""
https://github.com/tanimutomo/cifar10-c-eval
"""
import os
import torch
import models
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import loaders
import trainers
import transforms
import numpy as np
import pandas as pd


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default='work_dirs/datasets')
    parser.add_argument('--image_size', type=int, default=32,
                        help="input images are resized to this size")
    parser.add_argument('--log_dir', type=str, default='work_dirs/logs')
    parser.add_argument('--arch', type=str, default='resnet18small',
                        choices=models.MODEL_NAMES,
                        help="classifier architecture (default: resnet18small)")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_dirs', type=str)
    parser.add_argument('--rand_num_test', type=int, default=20,
                        help="number of random transformations per image to estimate the "
                             "expectation during testing")
    args = parser.parse_args()
    return args


def main(args):
    # load data and evaluate
    dataset = loaders.load_target_dataset(dataset=args.dataset,
                                          datadir=args.data_dir,
                                          image_size=args.image_size)
    args.num_classes = dataset['num_classes']
    test_set = dataset['test']
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # load model
    model = models.Classifier(args)
    model_dirs = args.model_dirs.split(',')
    model_dirs = [os.path.join(args.log_dir, model_dir) for model_dir in model_dirs]

    # variations
    transformations = ['identity', 'affine', 'rotate', 'perspective', 'crop',
                       'elastic_transform', 'fisheye', 'thin_plate_spline']

    correct_robs = dict()
    invariance_robs = dict()

    for model_dir in model_dirs:
        try:
            print("Evaluating model at path:", model_dir)
            ckpt = torch.load(model_dir, map_location='cpu')
            model.load_state_dict(ckpt['backbone'])  # new model
            # model = load_state(ckpt, model)
            model = model.to(args.device)
            model.eval()
        except Exception:
            print("Error while loading model at path:", model_dir)
            continue

        for transform in transformations:
            transformer = transforms.Transformer(transform, args.dataset)
            evaluator = trainers.create_evaluator(model, test_loader, transformer,
                                                  rand_num_test=args.rand_num_test,
                                                  device=args.device)
            stats = evaluator()
            correct_rob = stats['rob'] * 100  # percentage
            invariance_rob = stats['inv'] * 100
            correct_robs[transform] = correct_robs.get(transform, []) + [correct_rob]
            invariance_robs[transform] = invariance_robs.get(transform, []) + [invariance_rob]

    stats = pd.DataFrame({'correct': correct_robs, 'invariance': invariance_robs})
    stats = stats.applymap(lambda x: f"{np.array(x).mean():.2f} \u00B1 "
                                     f"{np.array(x).std():.2f}")
    print(stats)


if __name__ == "__main__":
    args = arg_parse()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
