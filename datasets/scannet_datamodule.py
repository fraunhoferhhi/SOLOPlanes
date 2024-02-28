import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from datasets.scannet_single import ScannetSingle
from datasets.scannet_multiview import ScannetMultiview

class ScannetDataModule(pl.LightningDataModule):
    def __init__( self, cfg,):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.shuffle = cfg.shuffle
        self.num_workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.mv_mode = cfg.model.mv_mode
        self.cfg = cfg
        self.train_subset = cfg.dataset.train_subset
        self.val_subset = cfg.dataset.val_subset
        self.collate_fn = ScannetMultiview.collate if self.mv_mode else ScannetSingle.collate

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.mv_mode:
                self.val_set = ScannetMultiview(self.cfg, 'val')
                self.train_set = ScannetMultiview(self.cfg, 'train')
            else:
                self.val_set = ScannetSingle(self.cfg, 'val')
                self.train_set = ScannetSingle(self.cfg, 'train')

            if type(self.train_subset) is float:
                assert self.train_subset < 1, "dataset subset should be either int or float < 1"
                self.train_subset = int(len(self.train_set) * self.train_subset)
                self.val_subset = int(len(self.val_set) * self.val_subset)
            if self.train_subset:
                train_indices = torch.randperm(len(self.train_set))[:self.train_subset]
                val_indices = torch.randperm(len(self.val_set))[:self.val_subset]
                self.train_set = Subset(self.train_set, train_indices)
                self.val_set = Subset(self.val_set, val_indices)

        if stage in (None, "test"):
            if self.mv_mode:
                self.test_set = ScannetMultiview(self.cfg, 'test')
            else:
                self.test_set = ScannetSingle(self.cfg, 'test')

    def train_dataloader(self):
        print(f"train set: {len(self.train_set)}")
        return DataLoader(
            self.train_set,
            self.batch_size,
            shuffle=self.shuffle,
            collate_fn = self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

    def val_dataloader(self):
        print(f"val set: {len(self.val_set)}")
        return DataLoader(
            self.val_set,
            self.batch_size,
            shuffle=False,
            collate_fn = self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

    def test_dataloader(self):
        print(f"test set: {len(self.test_set)}")
        return DataLoader(
            self.test_set,
            self.batch_size,
            shuffle=self.shuffle,
            collate_fn = self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

