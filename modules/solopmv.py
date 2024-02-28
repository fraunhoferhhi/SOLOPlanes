import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as m
from itertools import chain
from torch.nn.parallel import DataParallel
import pytorch_lightning as pl
from collections import defaultdict
from kornia.augmentation import Normalize

from .base.backbone import resnet18, resnet34, resnet50
from .pred_heads.solopv1_head import SOLOPv1Head
from .feat_heads.mask_feat_head import MaskFeatHead
from .feat_heads.plane_feat_head import PlaneFeatHead
from .base.fpn import FPN
from .pred_heads.plane_pred_head import PlanePredHead
from utils.plane_utils import *
from utils.losses import consistency_loss
from modules.augment_data import DataAug
from modules.misc import *

import code


class SOLOP(pl.LightningModule):
    
    def __init__(self,
                 cfg,
                 init_weights=False,
                 mode='train'):
        super(SOLOP, self).__init__()
        
        neck_channels = cfg.neck.out_channels
        mask_feat_channels = cfg.model.mask_feat_channels
        plane_feat_channels = cfg.model.plane_feat_channels
        use_same_feats=cfg.model.use_same_feat_lvls,
        out_lvls = 4 if use_same_feats else 5

        if cfg.backbone.name == 'resnet18':
            self.backbone = resnet18(pretrained=True, loadpath = cfg.backbone.path)
            self.neck = FPN(in_channels=[64, 128, 256, 512],out_channels=neck_channels,start_level=0,num_outs=5,upsample_cfg=dict(mode='nearest'))
        elif cfg.backbone.name == 'resnet34':
            self.backbone = resnet34(pretrained=True, loadpath = cfg.backbone.path)
            self.neck = FPN(in_channels=[64, 128, 256, 512],out_channels=neck_channels,start_level=0,num_outs=5,upsample_cfg=dict(mode='nearest'))
        elif cfg.backbone.name == 'resnet50':
            self.backbone = resnet50(pretrained=cfg.backbone.use_pretrained, loadpath = cfg.backbone.path)
            self.neck = FPN(in_channels=[256, 512, 1024, 2048],out_channels=neck_channels,start_level=0,num_outs=out_lvls,upsample_cfg=dict(mode='nearest'))
        else:
            raise NotImplementedError

        self.mask_feat_head = MaskFeatHead(in_channels=neck_channels,
                            out_channels=mask_feat_channels,
                            start_level=0,
                            end_level=3,
                            num_classes=mask_feat_channels)
        self.bbox_head = SOLOPv1Head(num_classes=cfg.model.num_classes,
                            in_channels=neck_channels,
                            seg_feat_channels=neck_channels,
                            stacked_convs=2,
                            strides=[8, 32, 32],
                            scale_ranges=((1, 112), (112, 224), (224, 896)),
                            num_grids=[36, 24, 16],
                            use_same_feats=use_same_feats,
                            ins_out_channels=mask_feat_channels,
                            loss_ins_weight=cfg.model.dice_loss_weight
                        )

        self.plane_head = PlanePredHead(neck_channels, 
                            plane_feat_channels, 
                            cfg.model.plane_head_kernel, 
                            cfg.model.use_plane_feat_head,
                            cfg.model.planefeat_startlvl,
                            cfg.model.planefeat_endlvl,
                            cfg.model.plane_feat_xycoord)

        self.mode = mode

        self.use_consistency = cfg.model.consistency_loss
        self.cate_loss_weight = cfg.model.cate_loss_weight

        self.lr = cfg.lr
        self.start_lr = cfg.start_lr
        self.lr_diff = self.lr - self.start_lr
        self.warmup_steps = cfg.lr_warmup_steps
        if self.warmup_steps > 0 and mode=='train':
            print(f"using lr {self.start_lr} to {self.lr} for {self.warmup_steps} steps")
        self.optimizer = cfg.optimizer

        self.test_cfg = cfg.test_cfg 
        self.batch_size = cfg.batch_size
        self.normalize_im = Normalize(mean=torch.tensor(cfg.dataset.img_mean, dtype=torch.float, device=self.device), 
                std=torch.tensor(cfg.dataset.img_std, dtype=torch.float, device=self.device))
        self.augment_data = cfg.dataset.augment and mode=='train'
        if self.augment_data and mode=='train':
            print('using augmented data')
            self.data_aug_module = DataAug()

        if self.mode == 'train':
            self.backbone.train(mode=True)
        else:
            self.backbone.train(mode=True)
        
        if init_weights:
            print('initializing weights')
            self.init_weights() 

    def configure_optimizers(self): 
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            }
        }
    def init_weights(self):
        #fpn
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()
        
        #mask feature mask
        if isinstance(self.mask_feat_head, nn.Sequential):
            for m in self.mask_feat_head:
                m.init_weights()
        else:
            self.mask_feat_head.init_weights()

        self.bbox_head.init_weights()
        self.plane_head.init_weights()
    

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x
    
    def forward(self, img):
        x = self.extract_feat(img)
        plane_params, plane_feat = self.plane_head(x, return_feature=True)
        cate_pred, kernel_pred = self.bbox_head(x)
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
        return plane_params, plane_feat, mask_feat_pred, cate_pred, kernel_pred

    def calc_losses_singleview(self, batch):
        img = batch['image']
        depth = batch['depth']
        planes = batch['perpix_planes']
        inst_planes = batch['inst_planes']
        inst_masks = batch['inst_masks']
        nonplanar_mask = batch['nonplanar_mask']
        ranges = batch['kinv_xy1']
        bboxes =  batch['inst_bboxes']
        inst_labels = batch['inst_semantic_labels']

        img = self.normalize_im(img)

        plane_params, _,  mask_feat_pred, cate_pred, kernel_pred = self.forward(img)

        mask_cate_losses = self.bbox_head.loss(
            cate_pred, kernel_pred, mask_feat_pred, plane_params, inst_planes, bboxes, inst_labels, inst_masks)
        plane_losses = self.plane_head.get_losses(plane_params, planes, depth, nonplanar_mask, ranges, surface_normal_loss = True)
        loss= {**mask_cate_losses, **plane_losses}
        return loss

    def train_val_step(self, batch, train=False):
        warped_feats = []
        losses = defaultdict(lambda: torch.zeros(1, device=self.device))
        for view in range(len(batch)):
            images = batch[view]['image'] 
            depths = batch[view]['depth']
            planes = batch[view]['perpix_planes']
            inst_planes = batch[view]['inst_planes']
            inst_masks = batch[view]['inst_masks']
            nonplanar_masks = batch[view]['nonplanar_mask']
            ranges = batch[view]['kinv_xy1']
            bboxes =  batch[view]['inst_bboxes']
            inst_labels = batch[view]['inst_semantic_labels']
            intrinsics = batch[view]['intrinsics']
            cam_poses = batch[view]['cam_pose']
            
            if self.augment_data and train:
                images = self.data_aug_module(images)
            images = self.normalize_im(images)

            plane_params, plane_feat, mask_feat_pred, cate_pred, kernel_pred = self.forward(images)

            mask_cate_losses = self.bbox_head.loss(
                    cate_pred, kernel_pred, mask_feat_pred, plane_params, inst_planes, bboxes, inst_labels, inst_masks)
            plane_losses = self.plane_head.get_losses(plane_params, planes, depths, nonplanar_masks, ranges, surface_normal_loss = True)

            pred_shape = plane_params.shape[-2:]
            for dic in [mask_cate_losses, plane_losses]:
                for k,v in dic.items():
                    losses[k] = losses[k] + v

            if view==0:
                src_gt_planes = planes
                src_gt_depths = depths
                src_nonplanar_masks = nonplanar_masks
                src_ranges = ranges
                src_img = images
                src_plane_pred = plane_params

                scale = pred_shape[-1] / images.shape[-1]
                source_intr_scaled = intrinsics
                source_intr_scaled[:,0:2,0:3] = source_intr_scaled[:,0:2,0:3] * scale
                source_depths_scaled = F.interpolate(depths, size=pred_shape)
                source_poses = cam_poses

                
            if view > 0:
                transforms = get_relative_transforms_batch(source_poses, cam_poses)
                us,vs, outprojection_mask = get_sampling_grid(source_depths_scaled, transforms, source_intr_scaled, pred_shape[0], pred_shape[1])
                warped_feat = grid_sampler(us,vs,plane_feat)
                warped_plane_pred = self.plane_head.get_plane_pred(warped_feat)
                normals, offsets = get_plane_params(warped_plane_pred, norm_axis=1)
                transforms21 = get_relative_transforms_batch(cam_poses, source_poses)
                warped_plane_pred = warp_planes(normals, offsets, transforms21)
                if self.use_consistency:
                    pl_consistency_loss = consistency_loss(src_plane_pred, warped_plane_pred, outprojection_mask.unsqueeze(1).bool())
                    losses["consistency_loss"] += pl_consistency_loss
                outprojection_mask = F.interpolate(outprojection_mask.unsqueeze(1).float(), nonplanar_masks.shape[-2:])
                warped_plane_losses = self.plane_head.get_losses(warped_plane_pred, src_gt_planes, src_gt_depths, src_nonplanar_masks, src_ranges, 
                        surface_normal_loss=True, 
                        outprojection_mask=outprojection_mask.bool())
                for k,v in warped_plane_losses.items():
                    losses[k] = losses[k] + v

        return losses

    def training_step(self, batch, batch_idx):
        if self.global_step <= self.warmup_steps:
            self.optimizers().param_groups[0]['lr'] = self.start_lr +  (self.global_step / self.warmup_steps) *  self.lr_diff
        loss = self.train_val_step(batch, train=True)
        losses = loss['loss_ins'] + loss['loss_cate'] + loss['l1_loss'] + loss['depth_loss'] + loss['q_loss'] + loss['surface_normal_loss']
        self.log("total", losses, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("mask loss", loss['loss_ins'], on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("cate loss", loss['loss_cate'], on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("l1 loss", loss['l1_loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("depth loss", loss['depth_loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("q loss", loss['q_loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("cossim loss", loss['surface_normal_loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return losses

    def validation_step(self, batch, batch_idx):
        loss = self.train_val_step(batch)
        losses = loss['loss_ins'] + loss['loss_cate'] + loss['l1_loss'] + loss['depth_loss'] + loss['q_loss'] + loss['surface_normal_loss']
        self.log("val_loss", losses, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return losses

    def predict(self, img):
        img = self.normalize_im(img)
        x = self.extract_feat(img)
        plane_params = self.plane_head(x)
        cate_pred, kernel_pred = self.bbox_head(x, eval=True)
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
        seg_result = self.bbox_head.get_seg(cate_pred, kernel_pred, mask_feat_pred, self.test_cfg, img.shape[-2:])
        plane_params = F.interpolate(plane_params, img.shape[-2:])
        return seg_result, plane_params

    def predict_fps(self, img):
        x = self.extract_feat(img)
        plane_params = self.plane_head(x)
        cate_pred, kernel_pred = self.bbox_head(x, eval=True)
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])
        seg = self.bbox_head.get_seg(cate_pred, kernel_pred, mask_feat_pred, self.test_cfg, img.shape[-2:])
        planes = F.interpolate(plane_params, img.shape[-2:])
        seg_masks, soft_masks, cate_label, cate_scores = tuple(map(list, zip(*seg)))
        inst_planes, inst_perpix_planes = multi_apply(inst_plane_pooling, soft_masks, seg_masks, planes)
        return seg_masks, torch.stack(inst_perpix_planes)
