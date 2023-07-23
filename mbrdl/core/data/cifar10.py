from torch.utils.data import Dataset
import torchvision as tv
import torch
from data.transforms import *



class CIFAR10Dataset(Dataset):
    def __init__(self, mode, challenge, dom, args, return_labels=True):
        """SVHN/GTSRB dataset for training models and classifiers.

        Params:
            mode: Determines kind of data that is returned.
                - choices:  'train' | 'test'.
            challenge: Kind of natural variation.
                - choices: 'brightness' | 'contrast' | 'contrast+brightness'
            dom: Domain to be used.
                - choices: 'low' | 'medium' | 'high'
            args: Command line arguments.
            return_labels: If True, returns image and label.  Otherwise
                only returns image (without label).
        """

        self._mode = mode
        self._challenge = challenge
        self._dom = dom
        self._return_labels = return_labels

        transform = tv.transforms.Compose([
            tv.transforms.Resize((args.data_size, args.data_size)),
            tv.transforms.ToTensor(),
            # tv.transforms.Normalize(torch.tensor([0.4376, 0.4437, 0.4728]), torch.tensor([0.1980, 0.2010, 0.1970])),
        ])

        split = False if 'test' in self._mode else True
        self._data = tv.datasets.CIFAR10(args.train_data_dir, train=split, transform=transform, download=False)
        self._data = self.change_data()

    def __getitem__(self, index: int):

        img, label = self._data[index]
        if self._return_labels is True:
            return img, label
        return img

    def __len__(self) -> int:
        """Returns number of datapoints in dataset."""

        return len(self._data)

    def change_data(self):
        if self._dom == 'orig':
            return self._data
        elif self._dom in ('randaugment', 'rotate'):
            new_data = []
            T = Transformer(self._dom)
            for i in range(len(self._data)):
                img, label = self._data[i]
                new_img = torch.squeeze(T.apply(torch.unsqueeze(img, 0)), 0)
                new_data.append((new_img, label))
        else:
            raise ValueError('Invalid dom.')
        return new_data

