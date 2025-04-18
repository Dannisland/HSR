import random

import math
import os
import pdb

import mmcv
import numpy as np
from mmengine import list_from_file
from torch.utils.data import Dataset
import dataset.transform as Xform
import torch


class DIV2K(Dataset):

    def __init__(self, data_list=None, training=False, cfg=None):
        super(DIV2K, self).__init__()
        self.cfg = cfg
        self.imgs = list_from_file(data_list, prefix=cfg.data_root + '/')
        assert self.cfg.patch_size % self.cfg.base_resolution == 0, "Patch size must base resolution"
        self.training = training

    def __len__(self):
        if not self.cfg.debug:
            return len(self.imgs) * self.cfg.loop
        else:
            return 512

    def __getitem__(self, index_long):
        """
        :param index_long:
        :return: RGB, np.float32
        """
        index = index_long % len(self.imgs)
        index_sec = random.randint(0, len(self.imgs) - 1)
        img_gt = mmcv.imread(self.imgs[index]).astype(np.float32) / 255.
        img_sec = mmcv.imread(self.imgs[index_sec]).astype(np.float32) / 255.
        img_sec_2 = mmcv.imread(self.imgs[random.randint(0, len(self.imgs) - 1)]).astype(np.float32) / 255.

        # if self.training:
        img_gt = Xform.random_crop(img_gt, self.cfg.patch_size)
        img_gt = Xform.augment(img_gt, hflip=self.cfg.hflip, rotation=self.cfg.rotation)
        img_sec = Xform.random_crop(img_sec, int(self.cfg.patch_size / 4))
        img_sec = Xform.augment(img_sec, hflip=self.cfg.hflip, rotation=self.cfg.rotation)
        img_sec_2 = Xform.random_crop(img_sec_2, int(self.cfg.patch_size / 2))
        img_sec_2 = Xform.augment(img_sec_2, hflip=self.cfg.hflip, rotation=self.cfg.rotation)

        img_gt = mmcv.bgr2rgb(img_gt)
        img_gt = torch.from_numpy(img_gt.transpose((2, 0, 1)))
        img_sec = mmcv.bgr2rgb(img_sec)
        img_sec = torch.from_numpy(img_sec.transpose((2, 0, 1)))
        img_sec_2 = mmcv.bgr2rgb(img_sec_2)
        img_sec_2 = torch.from_numpy(img_sec_2.transpose((2, 0, 1)))
        # return img_gt
        return {'img_gt': img_gt, 'img_sec': img_sec, 'img_sec_2': img_sec_2}


