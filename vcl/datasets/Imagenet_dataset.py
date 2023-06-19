import os
import cv2
import lmdb
import random
import numpy as np

import torch
import torch.utils.data as data

import mmcv
import os.path as osp
import glob

from .pipelines import RandomResizedCrop, Normalize

class ImageFolderLMDB(data.Dataset):
    def __init__(self, lmdb_path, im_size):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.im_size = im_size
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        self.normalize = Normalize(**img_norm_cfg)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(index).encode())

        # load img
        img_np = np.frombuffer(byteflow, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # img = img.astype(np.float32) / 255.

        # padding if image is too small
        h, w, _ = img.shape
        h_pad = max(0, self.im_size - h)
        w_pad = max(0, self.im_size - w)
        if h_pad != 0 or w_pad != 0:
            img = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

        # random crop
        h, w, _ = img.shape
        top = random.randint(0, h - self.im_size)
        left = random.randint(0, w - self.im_size)
        img = img[top:top + self.im_size, left:left + self.im_size, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.normalize(dict(imgs=[img], modality='RGB'))['imgs'][0]


        img = torch.from_numpy(img.transpose(2, 0, 1))

        return img

    def __len__(self):
        return 1281167