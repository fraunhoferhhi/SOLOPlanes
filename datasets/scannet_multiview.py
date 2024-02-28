import numpy as np
import cv2
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets.scannet_base import ScannetBase
import code


class ScannetMultiview(ScannetBase):
    def __init__(self, 
            cfg,
            mode: str = 'train', 
            *args):
        super().__init__(cfg, mode)
        self.n_views = cfg.dataset.mv.views
        self.steps_between_views = cfg.dataset.mv.steps_between_views
        self.fill_sceneimg_indices_precalc(mode)


    def _fill_sceneimg_indices(self, scene_id, num_images):
        return [[scene_id, i] for i in range(num_images) if i + self.steps_between_views*(self.n_views-1)  < num_images]

    def get_mv_sample(self, index, sceneimgidx=None):
        scene_id, img_idx = self.scene_image_indices[index]
        if sceneimgidx is not None:
            scene_id, img_idx =  sceneimgidx 
        mv_samples = []
        for i in range(0, self.n_views*self.steps_between_views, self.steps_between_views):
            sample = self.get_sample(scene_id, img_idx + i)
            if sample is None:
                new_idx = np.random.randint(0, len(self.scene_image_indices))
                return self.get_mv_sample(new_idx)
            mv_samples.append(sample)
        return mv_samples

    def __getitem__(self, index, sceneimgidx=None):
        return self.get_mv_sample(index, sceneimgidx)

    @staticmethod
    def collate(batch):
        result = []
        nv = len(batch[0])
        views = list(zip(*batch))
        for i in range(nv):
            collated_batch = ScannetBase._collate(views[i])
            result.append(collated_batch)
        return result

