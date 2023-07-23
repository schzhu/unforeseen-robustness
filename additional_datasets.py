"""
    A set of additional dataloaders
"""
import json
import os
import random
import re
from typing import Any, Callable, Optional, Tuple, cast
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from multi_illumination import multilum


# ----------------------------------------------------------------
#   Objectron dataset helpers
# ----------------------------------------------------------------
def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith( \
        extensions if isinstance(extensions, str) else tuple(extensions))


def _find_classes(root):
    """
        Customized version of find classes
    """
    classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {root}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def _sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def _make_dataset(root, class_to_idx, extensions, is_valid_file, nview):
    """
        Customized version of make_datset function.
    """
    root = os.path.expanduser(root)

    if class_to_idx is None:
        _, class_to_idx = _find_classes(os.path.join(root, 'frames'))
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    # cast the type
    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    # data-holders
    videos = []
    frames = {}
    metadata = {}

    available_classes = set()

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]

        # : frame dir
        frame_dir = os.path.join(root, 'frames', target_class)
        minfo_dir = os.path.join(root, 'bboxes', target_class)

        # : extract the frame information
        if not os.path.isdir(frame_dir): continue

        # : loop over it
        for vname in tqdm(sorted(os.listdir(frame_dir)), \
                          desc="[objectron-load-frames:{}]".format(target_class)):
            if 'batch-' not in vname: continue

            # :: skip if there is not enough frames
            nframe = len(os.listdir(os.path.join(frame_dir, vname)))
            if nframe < nview: continue

            # :: put the vtoken to a holder
            vtoken = '{}-{}'.format(target_class, vname)
            videos.append(vtoken)

            # :: create the space
            if vtoken not in frames: frames[vtoken] = []

            # :: loop over the frames in vname
            for fname in _sorted_alphanumeric(
                    os.listdir(os.path.join(frame_dir, vname))):  # sort by natural index
                # > sanity check
                if not fname.startswith('frame-'): continue

                # > collect
                fpath = os.path.join(frame_dir, vname, fname)
                if is_valid_file(fpath):
                    fitem = fpath, class_index
                    frames[vtoken].append(fitem)

                    # > to check the available classes
                    if target_class not in available_classes:
                        available_classes.add(target_class)

        # : loop over the metadata
        for vname in tqdm(sorted(os.listdir(minfo_dir)), \
                          desc="[objectron-load-minfos:{}]".format(target_class)):
            if 'batch-' not in vname: continue

            # :: create the video token (+ sanity check)
            vtoken = '{}-{}'.format(target_class, vname)
            if vtoken not in videos: continue

            # :: set the filename
            mfile = os.path.join(minfo_dir, vname, 'bounding_boxes.npy')
            if not os.path.exists(mfile): continue

            # :: load the metadata and store
            metadata[vtoken] = np.load(mfile)

    # sanity checks
    assert set(frames.keys()) == set(metadata.keys()), \
        ("Error: the names of videos aren't equal to the names of metadata, abort")

    # sanity checks
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return videos, frames, metadata


class Objectron(Dataset):
    """
        load the dataset in the following format.
        : datasets/objectron/preprocess
            -> train -> images      -> class -> video -> 64x64 png files
                     -> annotations -> class -> video -> textfiles
            -> test  -> images      -> class -> video -> 64x64 png files
                     -> annotations -> class -> video -> textfiles
        [Note: using this format, we can load by using PyTorch dataloaders.]
        randomly load 2 extra pngs from {01.png, ..., 23.png}
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            # target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            train: bool = True,
            nview: int = 1):

        # local variables
        self.split = 'train' if train else 'test'
        self.root = os.path.join(root, self.split)  # assume root is datasets/objectron/preprocess
        self.extensions = ('.png', '.npy')

        # construct dataset (only with file locations)
        self.classes, self.class_to_idx = _find_classes(os.path.join(self.root, 'frames'))
        self.videos, self.frames, self.metadata = _make_dataset( \
            self.root, self.class_to_idx, self.extensions, is_valid_file, nview)

        # store others
        self.loader = default_loader
        self.n_views = nview
        self.transform = transform

        # hyperparameters
        # self.frames_limit = 1000
        self.frames_limit = 20

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
            dict: {
            "anchor_view": tensor,
            "anchor_meta": tensor,
            "extra_views": list of tensors,
            "extra_metas": list of tensors,
            "last_visited": -1,
            "target": tensor,
            }
        """
        # choose the video (= a sample)
        video = self.videos[index]
        frames = self.frames[video]
        m_info = self.metadata[video]

        # extract the anchor frame index
        # anchor_fidx = len(frames) // 2
        anchor_fidx = random.randint(0, len(frames) - 1)
        anchor_fpath, target = frames[anchor_fidx]
        # anchor_fname = anchor_fpath.split('/')[-1]
        # anchor_fidx  = int(anchor_fname.split('-')[1])

        # ----------------------------------------
        #   load the anchor frame and metadata
        anchor_view = self.loader(anchor_fpath)

        # conventional supervised training
        if self.n_views == 1:
            if self.transform is not None:
                anchor_view = self.transform(anchor_view)
            return anchor_view, target

        # ours
        else:
            # data-holder
            output = {}
            # tensorize the PIL image to send to batched transform
            # can only be placed here instead of dataloader
            output['anchor_view'] = (T.ToTensor()(anchor_view) * 255).to(torch.uint8).unsqueeze(0)
            output['anchor_meta'] = torch.from_numpy(m_info[anchor_fidx]).float()
            output['anchor_meta'] = output['anchor_meta'].flatten()

            # ----------------------------------------
            #   load the other frame and metadata (multiple)
            output["extra_views"] = []
            output["extra_metas"] = []
            output["last_visited"] = -1
            output["target"] = target

            # : then choose the frames to inject
            # n_adjacent_frames = min(self.frames_limit, anchor_fidx)

            frame_idxs = list(range(max(anchor_fidx - self.frames_limit, 0),
                                    min(anchor_fidx + self.frames_limit, len(frames))))
            # frame_idxs = list(range(len(frames)))
            frame_idxs.remove(anchor_fidx)
            # frame_rand = np.random.choice(frame_idxs, self.n_views-1, replace=False)    # exclude anchor
            frame_rand = np.random.permutation(frame_idxs)  # catch loading errors
            # : construct
            for fidx in frame_rand:
                # :: load
                try:
                    cur_view = self.loader(frames[fidx][0])
                    # if self.transform is not None:
                    #     cur_view = self.transform(cur_view)
                    cur_meta = torch.from_numpy(m_info[fidx]).float()
                    cur_meta = cur_meta.flatten()
                except Exception:
                    continue

                # :: append
                output["extra_views"].append(
                    (T.ToTensor()(cur_view) * 255).to(torch.uint8).unsqueeze(0))
                output["extra_metas"].append(cur_meta)
                if len(output["extra_metas"]) == self.n_views - 1:
                    break

            if self.transform is not None:
                samples_arr = torch.cat([output["anchor_view"], *output["extra_views"]])
                samples_arr = self.transform(samples_arr)
                samples_list = torch.unbind(samples_arr)
                output["anchor_view"] = samples_list[0]
                output["extra_views"] = samples_list[1:]

            return output, target
        # end if...

    def __len__(self) -> int:
        return len(self.videos)


class Illumination(Dataset):
    def __init__(self, root, transform=None):
        multilum.set_datapath(root)
        self.transform = transform
        self.scenes = multilum.query_scenes()
        self.num_dirs = 25

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int):
        imgs = multilum.query_images(self.scenes[index], mip=5)
        imgs = torch.squeeze(torch.from_numpy(imgs)).permute(0, 3, 1, 2)
        imgs = imgs / 255
        output = {}
        if self.transform is not None:
            imgs = self.transform(imgs)
        output["anchor_view"] = imgs[0]
        output["extra_views"] = imgs[1:]
        return output, index


class CIFAR10_1(Dataset):
    def __init__(self, data_path, transform=None):
        self.images = np.load(os.path.join(data_path, 'cifar10.1_v6_data.npy'))
        self.labels = np.load(os.path.join(data_path, 'cifar10.1_v6_labels.npy'))
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CIFAR10_2(Dataset):
    def __init__(self, data_path, transform=None):
        self.train = np.load(os.path.join(data_path, 'cifar102_train.npz'))
        self.test = np.load(os.path.join(data_path, 'cifar102_test.npz'))
        self.images = np.concatenate((self.train['images'], self.test['images']))
        self.labels = np.concatenate((self.train['labels'], self.test['labels']))
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CIFAR10_C(Dataset):
    def __init__(self, data_path, name, transform=None):
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
        assert name in corruptions
        self.image_path = os.path.join(data_path, name + '.npy')
        self.target_path = os.path.join(data_path, 'labels.npy')

        self.data = np.load(self.image_path)
        self.targets = np.load(self.target_path)
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)


"""
    Main (to test the dataset)
"""
if __name__ == "__main__":

    # dataset path
    dataroot = '../datasets/objectron/preprocess'
    datamstd = False

    # check the mean/std
    if datamstd:
        # : create a dataset
        dataset = Objectron(dataroot, transform=None, train=True)
        print(" : Create the objectron dataset")

        # : compute the mean and std
        tot_data = []
        for sample, _ in dataset:
            cur_data = np.array(sample["anchor_view"]) / 255.
            tot_data.append(cur_data)

        # : stack to compute the mean/std
        tot_data = np.stack(tot_data, axis=0)
        print(" : Total: {}".format(tot_data.shape))

        # : compute mean/std
        print(" : Mean: {} | Std.: {}".format( \
            np.mean(tot_data, axis=(0, 1, 2)), \
            np.std(tot_data, axis=(0, 1, 2))))

    else:
        # : create 3D dataset
        dataset = Objectron(dataroot, transform=None, train=True, nview=3)
        print(" : Create the objectron dataset")

        # : loop once
        for sample, _ in dataset:
            print(sample["anchor_view"])
            print(sample["anchor_meta"].shape)
            print(len(sample["extra_metas"]))
            print(sample["extra_metas"][0].shape)
            break

    print(" : Done.")
    # done.
