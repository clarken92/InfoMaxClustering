# Transformation for a single Numpy image
import numpy as np
import torch
import cv2


# These codes are provided by CoinCheung
# https://github.com/CoinCheung/fixmatch-pytorch/transform.py
# ========================================= #
class PadAndRandomCrop:
    # Input tensor is expected to have shape of (H, W, 3)
    def __init__(self, border=4, crop_size=(32, 32)):
        self.border = border
        self.crop_size = crop_size

    def __call__(self, im):
        borders = [(self.border, self.border), (self.border, self.border), (0, 0)]
        canvas = np.pad(im, borders, mode='reflect')
        H, W, C = canvas.shape
        h, w = self.crop_size
        dh, dw = max(0, H-h), max(0, W-w)
        sh, sw = np.random.randint(0, dh), np.random.randint(0, dw)
        out = canvas[sh:sh+h, sw:sw+w, :]
        return out


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im):
        if np.random.rand() < self.p:
            im = im[:, ::-1, :]
        return im


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        im = cv2.resize(im, self.size)
        return im


class Normalize:
    # Inputs are pixel values in range of [0, 255], channel order is 'rgb'
    def __init__(self, mean, std):
        self.mean = np.array(mean, np.float32).reshape(1, 1, -1)
        self.std = np.array(std, np.float32).reshape(1, 1, -1)

    def __call__(self, im):
        assert isinstance(im, np.ndarray), f"type(im)={type(im)}!"
        assert im.dtype.name == 'uint8', f"im.dtype.name={im.dtype.name}!"

        if len(im.shape) == 4:
            mean, std = self.mean[None, ...], self.std[None, ...]
        elif len(im.shape) == 3:
            mean, std = self.mean, self.std
        else:
            raise ValueError(f"'im' must be a 4D or 3D numpy array. Found im.shape={im.shape}!")

        im = im.astype(np.float32) / 255.
        im -= mean
        im /= std
        return im


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, im):
        if len(im.shape) == 4:
            return torch.from_numpy(im.transpose(0, 3, 1, 2))
        elif len(im.shape) == 3:
            return torch.from_numpy(im.transpose(2, 0, 1))
        else:
            raise ValueError(f"'im' must be a 4D or 3D numpy array. Found im.shape={im.shape}!")


class ToTensor_v2:
    def __init__(self):
        pass

    def __call__(self, im):
        if im.dtype.name == 'uint8':
            im = im.astype(np.float32) / 255.0

        if len(im.shape) == 4:
            return torch.from_numpy(im.transpose(0, 3, 1, 2))
        elif len(im.shape) == 3:
            return torch.from_numpy(im.transpose(2, 0, 1))
        else:
            raise ValueError(f"'im' must be a 4D or 3D numpy array. Found im.shape={im.shape}!")


class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, im):
        for op in self.ops:
            im = op(im)
        return im


# Borrow the code from RandAugment
class CutOut:
    def __init__(self, pad_size, mask_color=(0, 0, 0)):
        self.pad_size = pad_size
        self.mask_color = mask_color

    def __call__(self, im):
        assert im.dtype.name == 'uint8', f"im.dtype.name={im.dtype.name}!"
        mask = np.array(self.mask_color, dtype=np.uint8)

        H, W = im.shape[0], im.shape[1]
        rh, rw = np.random.random(2)

        pad = self.pad_size // 2
        ch, cw = int(rh * H), int(rw * W)
        x1, x2 = max(ch - pad, 0), min(ch + pad, H)
        y1, y2 = max(cw - pad, 0), min(cw + pad, W)
        out = im.copy()
        out[x1:x2, y1:y2, :] = mask

        return out


# Code is borrowed from "ContrastiveClustering_pytorch_Yunfan-Li"
class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample
# ========================================= #
