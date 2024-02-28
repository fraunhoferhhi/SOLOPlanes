import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.plane_utils import *

def consistency_loss(src_preds, warped_preds, outprojection_mask):
    l1 = nn.L1Loss()
    src_preds = src_preds * ~outprojection_mask
    warped_preds = warped_preds * ~outprojection_mask
    return l1(src_preds, warped_preds)

def depth_loss(pred_depth, gt_depth):
    l1 = nn.L1Loss()
    pred_depth = F.interpolate(pred_depth, gt_depth.shape[-2:])
    valid_mask = gt_depth!=0
    pred_depth = pred_depth*valid_mask
    gt_depth = gt_depth*valid_mask
    return l1(pred_depth, gt_depth)

def plane_loss(pred_planes, gt_planes,  nonplanar_mask, surface_normal_loss=False, outprojection_mask=None):
    '''L1 loss and cosine similarity loss for surface normals'''
    B, _, _, _ = pred_planes.shape
    l1 = nn.L1Loss()
    cossim_loss = 0
    if outprojection_mask is not None:
        invalid_mask = (nonplanar_mask == 1) | (outprojection_mask == 1)
    else:
        invalid_mask = nonplanar_mask
    plane_param = pred_planes*~invalid_mask + invalid_mask
    gt_params = gt_planes*~invalid_mask + invalid_mask
    plane_loss = l1(plane_param, gt_params)
    if surface_normal_loss:
        cos_sim = F.cosine_similarity(plane_param.reshape(B, 3,-1), gt_params.reshape(B, 3,-1))
        cossim_loss =  torch.mean(1 - cos_sim)
    return plane_loss, cossim_loss

def induced_depth_qloss(pred_planes, gt_depth, ranges, nd=None, edge_mask=None, outprojection_mask=None):
    '''plane induced depth and Q loss'''
    B, _, _, _ = pred_planes.shape
    depth_valid_region = gt_depth!=0
    if outprojection_mask is not None:
        valid_mask = depth_valid_region *~outprojection_mask
    else:
        valid_mask = depth_valid_region
    pred_planes = pred_planes*valid_mask
    gt_depth = gt_depth*valid_mask

    Q_im = ranges*gt_depth 
    Q = Q_im.reshape(B, 3,-1)
    if nd is None:
        n, d = get_plane_params(pred_planes, norm_axis=1)
    else:
        n, d = nd
    nqi = torch.sum(n.reshape(B, 3, -1)*Q, 1)
    q_loss_im = abs(nqi - d.reshape(B, -1))

    depth = get_depth_from_plane_param(ranges, n, d)
    depth = torch.clamp(depth, 1e-6, 10)
    depth_loss_im = abs(depth - gt_depth.squeeze())
    if edge_mask is not None:
        q_loss_weighted = q_loss_im * edge_mask.reshape(B, -1)
        q_loss = torch.mean(q_loss_weighted)
        depth_loss_weighted = depth_loss_im * edge_mask
        depth_loss = torch.mean(depth_loss_weighted)
    else:
        q_loss = torch.mean(q_loss_im)
        depth_loss = torch.mean(depth_loss_im)

    return depth_loss, q_loss
