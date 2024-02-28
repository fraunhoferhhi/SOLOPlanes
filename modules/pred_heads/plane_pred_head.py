import torch
import torch.nn as nn
import torch.nn.functional as F

from ..feat_heads.plane_feat_head import PlaneFeatHead
from utils.plane_utils import *
from utils.losses import *
import code

class PlanePredHead(nn.Module):
    def __init__(self, 
            fpn_channels, 
            plane_feat_channels,  
            kernel, 
            use_plane_feat_head = False , 
            planefeat_startlvl = 0, 
            planefeat_endlvl = 1,
            xycoord_planefeat = False):
        super(PlanePredHead, self).__init__()
        self.use_plane_feat_head = use_plane_feat_head
        if use_plane_feat_head:
            self.plane_feat_head = PlaneFeatHead(in_channels=fpn_channels,
                                out_channels=plane_feat_channels,
                                start_level=planefeat_startlvl,
                                end_level=planefeat_endlvl,
                                num_classes=plane_feat_channels,
                                xy_coord=xycoord_planefeat)
        pad = kernel // 2
        self.plane_head = nn.Conv2d(plane_feat_channels, 3,  kernel, padding=pad)

    def forward(self, x, return_feature=False):
        if self.use_plane_feat_head:
            feat = self.plane_feat_head(x[self.plane_feat_head.start_level:self.plane_feat_head.end_level + 1])
        else:
            feat = x[0]
        if return_feature:
            return self.plane_head(feat), feat
        return self.plane_head(feat)

    def get_plane_pred(self, plane_feat):
        return self.plane_head(plane_feat)


    def get_losses(self, pred_planes, gt_planes, gt_depth, nonplanar_mask, ranges, surface_normal_loss = False, nd=None, outprojection_mask=None, edge_mask=None):
        loss = dict()
        pred_planes = F.interpolate(pred_planes, gt_planes.shape[-2:])
        loss['l1_loss'], loss['surface_normal_loss'] = plane_loss(pred_planes, gt_planes, nonplanar_mask, surface_normal_loss, outprojection_mask)
        loss['depth_loss'], loss['q_loss'] = induced_depth_qloss(pred_planes, gt_depth, ranges, nd, edge_mask, outprojection_mask)
        return loss

    def init_weights(self):
        self.plane_feat_head.init_weights()
        nn.init.xavier_uniform_(self.plane_head.weight)


