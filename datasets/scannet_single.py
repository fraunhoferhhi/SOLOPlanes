import os
import sys
import numpy as np
import cv2
import torch
import pytorch_lightning as pl
from datasets.scannet_base import ScannetBase


class ScannetSingle(ScannetBase):
    def __init__(self, cfg, mode: str = 'train', *args):
        super().__init__(cfg, mode)
        print('initializing scannet single...')
        self.fill_sceneimg_indices_precalc(mode)

    def get_sceneimg_idx(self, index):
        return self.scene_image_indices[index]

    def _fill_sceneimg_indices(self, scene_id, num_images):
        return [[scene_id, i] for i in range(num_images)]


    def __getitem__(self, index, try_step=1, sceneimgidx=None):
        scene_id, img_idx = self.scene_image_indices[index]
        if sceneimgidx is not None:
            scene_id, img_idx =  sceneimgidx 
        sample = self.get_sample(scene_id, img_idx)
        if sample is None:
            new_idx = np.random.randint(0, len(self.scene_image_indices))
            return self.__getitem__(new_idx)
        return sample

    @staticmethod
    def collate(batch):
        return ScannetBase._collate(batch)

