"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf
from glob import glob


class ImageNet(datasets.ImageFolder):
    def __init__(self, root, split='train', split_prefix='ILSVRC2012_img_', transform=None):
        root = os.path.join(root, split_prefix + split)
        assert os.path.isdir(root), f"The root directory '{root}' does not exist!"

        super(ImageNet, self).__init__(root, transform=None)

        self.split = split
        self.transform = transform
        self._resize_fn = tf.Resize(256)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self._resize_fn(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self._resize_fn(img)
        return img


class ImageNet_w_Meta(ImageNet):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self._resize_fn(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out


class ImageNetSubset(data.Dataset):
    def __init__(self, root, subset_file, split='train',
                 split_prefix='ILSVRC2012_img_', transform=None):

        super(ImageNetSubset, self).__init__()

        root = os.path.join(root, split_prefix + split)
        assert os.path.isdir(root), f"The root directory '{root}' does not exist!"
        self.root = root

        self.split = split
        self.transform = transform

        assert os.path.isfile(subset_file), f"The subset file '{subset_file}' does not exist!"

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()

        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i))

        self.imgs = imgs
        self.classes = class_names
        self._resize_fn = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self._resize_fn(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self._resize_fn(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class ImageNetSubset_w_Meta(ImageNetSubset):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self._resize_fn(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}

        return out
