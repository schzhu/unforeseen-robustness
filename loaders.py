import os
import numpy as np

import torch
from torch.utils.data import random_split, Subset
from torchvision import transforms as T
from torchvision import datasets
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, STL10, CelebA, ImageNet, Caltech256

from additional_datasets import Objectron, CIFAR10_1, CIFAR10_2, CIFAR10_C, Illumination

SEED = 42

# mean and std of each dataset
STATS = {
    'cifar10': ([0.4914, 0.4821, 0.4465], [0.2468, 0.2433, 0.2613]),
    'cifar10a': ([0.4914, 0.4821, 0.4465], [0.2468, 0.2433, 0.2613]),
    'cifar10b': ([0.4914, 0.4821, 0.4465], [0.2468, 0.2433, 0.2613]),
    'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    'mnist': ([0.1309, 0.1309, 0.1309], [0.2893, 0.2893, 0.2893]),
    'svhn': ([0.4376, 0.4437, 0.4727], [0.1977, 0.2007, 0.1967]),
    'stl10': ([0.4407, 0.4274, 0.3859], [0.2501, 0.2430, 0.2513]),
    # 'tinyimagenet32': ([0.4807, 0.4485, 0.3980], [0.2538, 0.2453, 0.2600]),  # 32 x 32
    'tinyimagenet': ([0.4802, 0.4480, 0.3975], [0.2768, 0.2689, 0.2818]),
    # 64 x 64, stddev affected
    # 'objectron32':    ([0.5746, 0.5096, 0.4644], [0.2436, 0.2359, 0.2358]),  # 32 x 32
    'objectron': ([0.5745, 0.5096, 0.4645], [0.2517, 0.2438, 0.2432]),  # 64 x 64
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'celeba': ([0.5063, 0.4258, 0.3832], [0.2967, 0.2766, 0.2765]),
    'caltech256': ([0.5520, 0.5336, 0.5050], [0.2964, 0.2929, 0.3079]),
    'illumination': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

STATS = {k: (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in STATS.items()}


def seed_generator(seed):
    return torch.Generator().manual_seed(seed)


class CombinedDatasets(torch.utils.data.Dataset):
    def __init__(self, dataset_tar, dataset_src, mode='same_bs_rand_trunc', encore=0):
        self.dataset_tar = dataset_tar
        self.dataset_src = dataset_src
        self.mode = mode
        self._src_shuffle_map = torch.randperm(len(self.dataset_src),
                                               generator=seed_generator(SEED))
        self.encore = encore

    def get_index_src(self, i):
        if self.mode == 'same_bs_trunc_head':
            idx = i % len(self.dataset_src)  # only for len(src) < len(tar), which is MNIST
        elif self.mode == 'same_bs_rand_trunc':
            idx = self._src_shuffle_map[i % len(self.dataset_src)]
        elif self.mode == 'same_bs_iter_all':
            idx = self._src_shuffle_map[i % len(self.dataset_src)]
        elif self.mode == 'same_iters':
            raise 'Not implemented yet'
        else:
            raise Exception(f'Unknown datasets concat mode {self.mode}')
        return idx

    def reset_src_shuffle_map(self):
        self._src_shuffle_map = torch.randperm(len(self.dataset_src))

    def __getitem__(self, i):
        try:
            if self.encore == 0:
                return self.dataset_tar[i], self.dataset_src[self.get_index_src(i)]
            else:
                encore_samples = []
                indices = np.random.choice(len(self.dataset_tar), self.encore, replace=False)
                for j in indices:
                    encore_samples.append((self.dataset_tar[j],
                                           self.dataset_src[self.get_index_src(j)]))
                encore_samples.append((self.dataset_tar[i],
                                       self.dataset_src[self.get_index_src(i)]))
                return encore_samples

        except FileNotFoundError:
            print("Warning! FileNotFoundError")
            return None

    def __len__(self):
        return len(self.dataset_tar)


def load_target_dataset(dataset=None,
                        datadir=None,
                        augment=False,
                        image_size=32,
                        use_validation=False):
    if augment:
        augmenter = T.RandAugment(magnitude=9)
    else:
        augmenter = T.Lambda(identity_map)

    if dataset in ['r2n2', 'objectron']:  # augment r2n2 & objectron b/c special dataset __get_item
        to_tensor = T.Lambda(tiny_to_tensor)
    else:
        to_tensor = T.ToTensor()

    if dataset == 'cifar10':
        train_transform = T.Compose([T.Resize(image_size),
                                     augmenter,
                                     T.ToTensor(),
                                     T.Normalize(*STATS[dataset])])
        test_transform = T.Compose([T.Resize(image_size),
                                    T.ToTensor(),
                                    T.Normalize(*STATS[dataset])])
        train_val = CIFAR10(root=datadir, train=True, transform=train_transform, download=False)
        if use_validation:
            train, val = random_split(train_val, [45000, 5000], generator=seed_generator(SEED))
        else:
            train, val = train_val, None
        test = CIFAR10(root=datadir, train=False, transform=test_transform, download=False)
        num_classes = 10
    elif dataset == 'cifar100':
        train_transform = T.Compose([T.Resize(image_size),
                                     augmenter,
                                     T.ToTensor(),
                                     T.Normalize(*STATS[dataset])])
        test_transform = T.Compose([T.Resize(image_size),
                                    T.ToTensor(),
                                    T.Normalize(*STATS[dataset])])
        train_val = CIFAR100(root=datadir, train=True, transform=train_transform, download=True)
        if use_validation:
            train, val = random_split(train_val, [45000, 5000], generator=seed_generator(SEED))
        else:
            train, val = train_val, None
        test = CIFAR100(root=datadir, train=False, transform=test_transform, download=True)
        num_classes = 100

    elif dataset == 'cifar10a':
        train_transform = T.Compose([T.Resize(image_size),
                                     T.ToTensor(),
                                     T.Normalize(*STATS[dataset])])
        test_transform = T.Compose([T.Resize(image_size),
                                    T.ToTensor(),
                                    T.Normalize(*STATS[dataset])])
        train_val = CIFAR10(root=datadir, train=True, transform=train_transform, download=True)
        train_val, _ = random_split(train_val, [25000, 25000], generator=seed_generator(SEED))
        train, val = random_split(train_val, [22500, 2500], generator=seed_generator(SEED))
        test = CIFAR10(root=datadir, train=False, transform=test_transform, download=True)
        num_classes = 10

    elif dataset == 'tinyimagenet':
        train_transform = T.Compose([T.Resize(image_size),
                                     augmenter,
                                     to_tensor,
                                     T.Normalize(*STATS[dataset])])
        test_transform = T.Compose([T.Resize(image_size),
                                    to_tensor,
                                    # bug shouldn't use augment here # T.RandAugment(),
                                    T.Normalize(*STATS[dataset])])
        train_val = datasets.ImageFolder(root=os.path.join(datadir, 'tiny-imagenet-200/train'),
                                         transform=train_transform)
        train, val = random_split(train_val, [90000, 10000], generator=seed_generator(SEED))
        test = datasets.ImageFolder(root=os.path.join(datadir, 'tiny-imagenet-200/val/images'),
                                    transform=test_transform)
        num_classes = 200

    elif dataset == 'objectron':
        train_transform = T.Compose([T.Resize(image_size),
                                     augmenter,
                                     T.ToTensor(),
                                     T.Normalize(*STATS[dataset])])
        test_transform = T.Compose([T.Resize(image_size),
                                    T.ToTensor(),
                                    T.Normalize(*STATS[dataset])])
        train_val = Objectron(root=os.path.join(datadir, 'objectron/preprocess'),
                              transform=train_transform, train=True)
        if use_validation:
            train, val = random_split(train_val, [9331, 2333], generator=seed_generator(SEED))
        else:
            train, val = train_val, None
        test = Objectron(root=os.path.join(datadir, 'objectron/preprocess'),
                         transform=test_transform, train=False)
        num_classes = 9

    elif dataset == 'imagenet':
        train_transform = T.Compose([T.Resize(256),
                                     T.CenterCrop(image_size),
                                     T.ToTensor(),
                                     T.Normalize(*STATS[dataset])])
        test_transform = T.Compose([T.Resize(256),
                                    T.CenterCrop(image_size),
                                    T.ToTensor(),
                                    T.Normalize(*STATS[dataset])])

        # train_val = ImageNet(root=datadir, split='train', transform=train_transform)
        train_val = ImageNet(root=os.path.join(datadir, 'ImageNet/ILSVRC2012'), split='train',
                             transform=train_transform)

        # use a subset of data
        classes = train_val.classes[:10]
        class_indices = []
        for i in range(len(train_val)):
            if train_val.classes[train_val.targets[i]] in classes:
                class_indices.append(i)
        train_val = torch.utils.data.Subset(train_val, class_indices)

        train, val = train_val, None

        num_classes = 10

        # if use_validation:
        #     train, val = random_split(train_val, [896817, 384350], generator=seed_generator(SEED))
        # else:
        #     train, val = train_val, None

        # test = ImageNet(root=datadir, split='val', transform=test_transform)
        # num_classes = 1000
        test = train

    elif dataset == 'stl10':
        transform = T.Compose([T.Resize(image_size),
                               T.ToTensor(),
                               T.Normalize(*STATS[dataset])])
        train = STL10(datadir, split='train', transform=transform, download=False)
        test = STL10(datadir, split='test', transform=transform, download=False)
        train_val = train
        val = None
        num_classes = 10
    elif dataset == 'celeba':
        transform = T.Compose([T.Resize((image_size, image_size)),
                               T.ToTensor(),
                               T.Normalize(*STATS[dataset])])
        train = CelebA(datadir, split='train', transform=transform, download=False)
        test = CelebA(datadir, split='test', transform=transform, download=False)
        train = Subset(train, torch.randperm(len(train))[:10000])
        train_val = train
        val = None
        num_classes = 2

    else:
        raise Exception(f'Unknown dataset {dataset}')

    return dict(trainval=train_val,
                train=train,
                val=val,
                test=test,
                num_classes=num_classes)


def identity_map(x):
    return x


def convert_rgb(x):
    return x.convert('RGB')


def tiny_to_tensor(x):
    return x.float() / 255


def load_source_dataset(dataset=None,
                        datadir=None,
                        augment=False,
                        image_size=32,
                        rand_num_train=1,
                        train_size=0):
    # gray_mnist = T.Grayscale(num_output_channels=3)
    if dataset == 'mnist':
        to_rgb = T.Grayscale(num_output_channels=3)
    elif dataset == 'caltech256':
        to_rgb = T.Lambda(convert_rgb)
    else:
        to_rgb = T.Lambda(identity_map)

    if dataset in ['r2n2', 'objectron']:  # augment r2n2 & objectron b/c special dataset __get_item
        to_tensor = T.Lambda(tiny_to_tensor)
    else:
        to_tensor = T.ToTensor()

    if augment:
        augmenter = T.RandAugment(magnitude=9)
    else:
        augmenter = T.Lambda(identity_map)

    transform_train = T.Compose([T.Resize((image_size, image_size)),
                                 to_rgb,
                                 augmenter,
                                 to_tensor,
                                 T.Normalize(*STATS[dataset])])

    transform_test = T.Compose([T.Resize((image_size, image_size)),
                                to_rgb,
                                to_tensor,
                                T.Normalize(*STATS[dataset])])
    # if data_aug is None:
    # else:
    #     transform_train = T.Compose([T.Resize(image_size),
    #                                  to_rgb,
    #                                  T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
    #                                  T.ToTensor()])
    #     if data_aug == 'diff':
    #         transform_train = MultiView(transform_train)

    if dataset == 'cifar10':
        train = CIFAR10(root=datadir, train=True, transform=transform_train, download=True)
        test = CIFAR10(root=datadir, train=False, transform=transform_test, download=True)

    elif dataset == 'cifar10a':
        train = CIFAR10(root=datadir, train=True, transform=transform_train, download=True)
        train, _ = random_split(train, [25000, 25000], generator=seed_generator(SEED))
        test = CIFAR10(root=datadir, train=False, transform=transform_test, download=True)

    elif dataset == 'cifar10b':
        train = CIFAR10(root=datadir, train=True, transform=transform_train, download=True)
        _, train = random_split(train, [25000, 25000], generator=seed_generator(SEED))
        test = CIFAR10(root=datadir, train=False, transform=transform_test, download=True)

    elif dataset == 'mnist':
        train = MNIST(root=datadir, train=True, transform=transform_train, download=True)
        test = MNIST(root=datadir, train=False, transform=transform_test, download=True)

    elif dataset == 'svhn':
        train = SVHN(root=datadir, split='train', transform=transform_train, download=True)
        test = SVHN(root=datadir, split='test', transform=transform_test, download=True)

    elif dataset == 'cifar100':
        train = CIFAR100(root=datadir, train=True, transform=transform_train, download=True)
        test = CIFAR100(root=datadir, train=False, transform=transform_test, download=True)

    elif dataset == 'stl10':
        train = STL10(datadir, split='unlabeled', transform=transform_train, download=False)
        test = STL10(datadir, split='test', transform=transform_test, download=False)

    elif dataset == 'celeba':
        train = CelebA(datadir, split='train', transform=transform_train, download=True)
        test = CelebA(datadir, split='test', transform=transform_test, download=True)

    elif dataset == 'caltech256':
        train = Caltech256(datadir, transform=transform_train, download=False)
        test = Caltech256(datadir, transform=transform_test, download=False)
        test, _ = random_split(test, [10000, 20607], generator=seed_generator(SEED))

    elif dataset == 'tinyimagenet':
        train_val = datasets.ImageFolder(root=os.path.join(datadir, 'tiny-imagenet-200/train'),
                                         transform=transform_train)
        train, val = random_split(train_val, [90000, 10000], generator=seed_generator(SEED))
        test = datasets.ImageFolder(root=os.path.join(datadir, 'tiny-imagenet-200/val/images'),
                                    transform=transform_test)

    elif dataset == 'objectron':
        train = Objectron(root=os.path.join(datadir, 'objectron/preprocess'),
                          train=True, transform=transform_train, nview=rand_num_train + 1)
        test = Objectron(root=os.path.join(datadir, 'objectron/preprocess'),
                         train=False, transform=transform_test, nview=rand_num_train + 1)

    elif dataset == 'illumination':
        transform = T.Compose([T.CenterCrop(125),
                               T.Resize(image_size),
                               T.Normalize(*STATS[dataset])])

        train = Illumination(root=os.path.join(datadir, 'illumination'), transform=transform)
        test = train

    else:
        raise Exception(f'Unknown dataset {dataset}')
    if train_size > 0:
        if os.path.exists(os.path.join(datadir, 'temp', dataset+'_rand_indices')):
            indices = torch.load(os.path.join(datadir, 'temp', dataset+'_rand_indices'))
        else:
            indices = torch.randperm(len(train))
            torch.save(indices, os.path.join(datadir, 'temp', dataset+'_rand_indices'))
            print('Generate random indices for', dataset)

        train = Subset(train, indices[:train_size])

    return dict(train=train, test=test)


def load_dataset_combo(dataset_name_tar,
                       dataset_name_src,
                       data_dir_tar,
                       data_dir_src=None,
                       mode='same_bs_trunc_head',
                       aug_src=False,
                       aug_tar=False,
                       image_size=32,
                       rand_num_train=1,
                       encore=0,
                       use_validation=False,
                       src_size=0):
    """

    :param encore: spit out a list of tar-src pairs of length encore.
                   useful when training discriminator for 'encore' iterations
                   while training generator once.
    """
    dataset_tar = load_target_dataset(dataset=dataset_name_tar,
                                      datadir=data_dir_tar,
                                      image_size=image_size,
                                      augment=aug_tar,
                                      use_validation=use_validation)
    dataset_src = load_source_dataset(dataset=dataset_name_src,
                                      datadir=data_dir_src,
                                      augment=aug_src,
                                      image_size=image_size,
                                      rand_num_train=rand_num_train,
                                      train_size=src_size)
    dataset_combo_train = CombinedDatasets(dataset_tar['train'], dataset_src['train'],
                                           mode, encore)

    return dict(train_combo=dataset_combo_train,
                train_tar=dataset_tar['train'],
                train_src=dataset_src['train'],
                val_tar=dataset_tar['val'],
                test_tar=dataset_tar['test'],
                test_src=dataset_src['test'],
                num_classes=dataset_tar['num_classes'])


def load_ood_dataset(dataset=None, datadir=None, image_size=32, name=None):
    if dataset == 'cifar10_1':
        path = os.path.join(datadir, "cifar10_1")
        transform = T.Compose([T.ToTensor(), T.Normalize(*STATS['cifar10'])])
        test = CIFAR10_1(data_path=path, transform=transform)

    if dataset == 'cifar10_2':
        path = os.path.join(datadir, "cifar10_2")
        transform = T.Compose([T.ToTensor(), T.Normalize(*STATS['cifar10'])])
        test = CIFAR10_2(data_path=path, transform=transform)

    if dataset == 'cifar10-c':
        path = os.path.join(datadir, "cifar-10-c/CIFAR-10-C")
        transform = T.Compose([T.ToTensor(), T.Normalize(*STATS['cifar10'])])
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
        assert name in corruptions
        test = CIFAR10_C(data_path=path, name=name, transform=transform)

    return test


def collate_fn_filter_out_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
