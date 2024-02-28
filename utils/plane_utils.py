import sys
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
import matplotlib.transforms as mtrans
from config.solopmv_cfg import config as cfg
import cupy as cp
import code

# --------------------- data utils --------------------------
# https://github.com/leejaeyong7/planercnn/blob/51fab9328e6bb794211d00bc8ea21db50d2e6e62/utils.py
def calcPlaneDepths(planes, width, height, intrinsics, max_depth=cfg.dataset.MAX_DEPTH):
    fx = intrinsics[0,0]
    fy = intrinsics[1,1]
    mx = intrinsics[0, 2]
    my = intrinsics[1, 2]
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    urange = (x - mx) / fx
    vrange = (y - my) / fy
    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
    
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
    return planeDepths


def cleanSegmentation(image, planes, plane_info, segmentation, depth, intrinsics, planeAreaThreshold=200, planeWidthThreshold=10, depthDiffThreshold=cfg.dataset.depth_error_margin, validAreaThreshold=0.5, brightThreshold=20, confident_labels={}, return_plane_depths=False):
    planeDepths = calcPlaneDepths(planes, segmentation.shape[1], segmentation.shape[0], intrinsics).transpose((2, 0, 1))
    newSegmentation = np.full(segmentation.shape, fill_value=-1)
    try:
        validMask = np.logical_and(np.linalg.norm(image, axis=-1) > brightThreshold, depth > 1e-4)
        depthDiffMask = np.logical_or(np.abs(planeDepths - depth) < depthDiffThreshold, depth < 1e-4)
    except Exception as e:
        print(e)
        return None, None

    for segmentIndex in np.unique(segmentation):
        if segmentIndex < 0:
            continue
        segmentMask = segmentation == segmentIndex

        try:
            plane_info[segmentIndex][0][1]
        except:
            print('invalid plane info')
            return None, None
        if plane_info[segmentIndex][0][1] in confident_labels:
            if segmentMask.sum() > planeAreaThreshold:
                newSegmentation[segmentMask] = segmentIndex
                pass
            continue
        oriArea = segmentMask.sum()
        segmentMask = np.logical_and(segmentMask, depthDiffMask[segmentIndex])
        newArea = np.logical_and(segmentMask, validMask).sum()
        if newArea < oriArea * validAreaThreshold:
            continue
        segmentMask = segmentMask.astype(np.uint8)
        segmentMask = cv2.dilate(segmentMask, np.ones((3, 3)))
        numLabels, components = cv2.connectedComponents(segmentMask)
        for label in range(1, numLabels):
            mask = components == label
            ys, xs = mask.nonzero()
            area = float(len(xs))
            if area < planeAreaThreshold:
                continue
            size_y = ys.max() - ys.min() + 1
            size_x = xs.max() - xs.min() + 1
            length = np.linalg.norm([size_x, size_y])
            if area / length < planeWidthThreshold:
                continue
            newSegmentation[mask] = segmentIndex
            continue
        continue
    if return_plane_depths:
        return newSegmentation, planeDepths
    return newSegmentation

def transformPlanes(transformation, planes):
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    centers = planes # K, 3
    centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1) # K, 4
    newCenters = np.transpose(np.matmul(transformation, np.transpose(centers))) # K, 4
    newCenters = newCenters[:, :3] / newCenters[:, 3:4] # K, 3

    refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
    refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
    newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
    newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

    planeNormals = newRefPoints - newCenters
    planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
    planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
    newPlanes = planeNormals * planeOffsets
    return newPlanes.astype(np.float32)

def process_gt_planes(image, planes, segmentation, plane_info, depth, cam_pose, intrinsics):
    # https://github.com/leejaeyong7/planercnn/blob/master/datasets/scannet_scene.py
    # convert segmentation rgb img to idx img and prune planes
    extrinsics = np.linalg.inv(cam_pose)
    extrinsics[[1,2], :] = extrinsics[[2,1],:]
    extrinsics[2] *= -1
    segmentation = (segmentation[:, :,2] * 256 * 256 + segmentation[:, :,1] * 256 + segmentation[:, :, 0]) // 100 - 1

    segments, counts = np.unique(segmentation, return_counts=True)
    if segments.min() < -1:
        logging.warn(
            f"weird segment -- segments max min {segments.max()} {segments.min()}")
    segmentList = zip(segments.tolist(), counts.tolist())
    segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]
    # unique segmentation idx plane params & count in tuple
    segmentList = sorted(segmentList, key=lambda x: -x[1])
    newPlanes = []
    newPlaneInfo = []
    newSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
    newIndex = 0
    # remove invalid planes
    for oriIndex, count in segmentList:
        if count < 500:
            continue
        if oriIndex >= len(planes):
            continue
        if np.linalg.norm(planes[oriIndex]) < 1e-4:
            continue
        newPlanes.append(planes[oriIndex])
        newSegmentation[segmentation == oriIndex] = newIndex
        newPlaneInfo.append(plane_info[oriIndex] + [oriIndex])
        newIndex += 1
        continue

    segmentation = newSegmentation
    planes = np.array(newPlanes)
    plane_info = newPlaneInfo  

    if len(planes) > 0:
        planes = transformPlanes(extrinsics, planes)
        segmentation, plane_depths = cleanSegmentation(image, planes, plane_info, segmentation, depth, intrinsics, return_plane_depths=True)
        if segmentation is None:
            return None, None, None
        plane_depths = calcPlaneDepths(planes, segmentation.shape[1], segmentation.shape[0], intrinsics).transpose((2, 0, 1))

        masks = (np.expand_dims(segmentation, -1) == np.arange(len(planes))).astype(np.float32)
        plane_depth = (plane_depths.transpose((1, 2, 0)) * masks).sum(2)
        plane_mask = masks.max(2)
        plane_mask *= (depth > 1e-4).astype(np.float32)            
        plane_area = plane_mask.sum()
        depth_error = (np.abs(plane_depth - depth) * plane_mask).sum() / max(plane_area, 1)
        if depth_error > cfg.dataset.depth_error_margin:
            print('depth error', depth_error)
            planes = []

    return segmentation, planes, plane_info

# --------------------- plane utils --------------------------
def load_planes(planes_path, gt_plane_info_path=None):
    try:
        planes = np.load(planes_path, allow_pickle=True).astype(np.float32)
        if gt_plane_info_path:
            gt_plane_info = np.load(gt_plane_info_path, allow_pickle=True)
            return planes, gt_plane_info
    except Exception as e:
        print(f"exception in load_planes: {e}")
        return None
    return planes

def get_imgtocamcoord_batch(intrinsics, height, width):
    """
    Params
        intrinsics: [B, 4, 4]
    """
    b_size = intrinsics.shape[0]
    x, y = torch.meshgrid(torch.arange(width, device=intrinsics.device), torch.arange(height, device=intrinsics.device), indexing='xy')
    xs = x[None,:,:].repeat_interleave(b_size, axis=0)
    ys = y[None,:,:].repeat_interleave(b_size, axis=0)
    urange = (xs.reshape(b_size, -1) - intrinsics[:,0,2, None]) / intrinsics[:,0,0, None]
    vrange = (ys.reshape(b_size, -1) - intrinsics[:,1,2, None]) / intrinsics[:,1,1, None]
    kinvxy1 = torch.stack([urange, vrange, torch.ones(urange.shape, device=urange.device)], axis=1) # B, 3, N
    return kinvxy1

def get_ranges(intrinsics, height, width):
    rpy = np if intrinsics.device=="cpu" else cp
    intrinsics = rpy.asarray(intrinsics.detach())
    x, y = rpy.meshgrid(rpy.arange(width), rpy.arange(height), indexing='xy')
    urange = (x - intrinsics[0,2]) / intrinsics[0,0]
    vrange = (y - intrinsics[1,2]) / intrinsics[1,1]
    ranges = rpy.stack([urange, vrange, rpy.ones(urange.shape)], axis=0) # 3, h, w
    return ranges

def get_relative_transforms_batch(pose1, pose2):
    """
    scannet pose is camera to world
    Params
        pose: Bx4x4

    """
    transform12 = torch.bmm(torch.linalg.inv(pose2), pose1)
    return transform12

def convert_to_nd_planeparams(mu):
    """
    Params
        mu: B, C, H, W
    """
    d = torch.linalg.norm(mu, axis=1, keepdims=True) # b, 1, h, w
    offsets = 1/d
    normals = mu * offsets
    mu_converted = normals*offsets
    return mu_converted, normals, offsets

def get_plane_params(planes, norm_axis=None):
    """
    planes are n*d format. flexible norm dimension. 
    """
    if norm_axis is None:
        norm_axis = planes.shape.index(3)
    offsets = torch.linalg.norm(planes, axis=norm_axis, keepdims=True) # b, 1, h, w
    normals = planes / torch.maximum(offsets, torch.tensor(1e-8))
    return normals, offsets # b, c, h, w

def get_instance_param(soft_mask, perpix_planes):
    weighted_planes = soft_mask * perpix_planes
    inst_param = weighted_planes.reshape(3, -1).sum(1) / (soft_mask.sum() + 1e-8)
    return inst_param

def inst_plane_pooling(soft_masks, seg_masks, pred_planes):
    """
    Params:
        soft_masks, seg_masks: 1, H, W
        pred_planes: 3, H, W
    Returns
        inst_planes, inst_perpix_planes: 3, H, W
    """
    inst_planes = torch.zeros(pred_planes.shape, device=pred_planes.device)
    inst_perpix_planes = pred_planes.clone()
    for idx,m in enumerate(seg_masks):
        cur_msk = soft_masks[idx] * seg_masks[idx]
        inst_p = get_instance_param(cur_msk, pred_planes) 
        inst_planes[:, m] = inst_p.unsqueeze(1)
        inst_perpix_planes[:,m] = inst_p.unsqueeze(1)
    return inst_planes, inst_perpix_planes

def warp_planes(normal1, offset1, transform12):
    """
    Warp planes from cam1 -> cam2
    Params:
        normals: B, 3, H, W
        offsets: B, H, W
        transform12: 4, 4
    Return:
        planes12: 125, C, H, W
    """
    b, c, h, w = normal1.shape
    R12 = transform12[:, :3, :3]
    t12 = transform12[:, :-1, -1]
    normal1 = normal1.reshape(b, c,-1)
    normal12 = torch.bmm(R12, normal1) 
    offset12 = offset1.reshape(b,-1) + torch.sum(normal1*t12.unsqueeze(-1), dim=1)
    planes12 = (normal12 * offset12.unsqueeze(1)).reshape(b, c, h, w)
    return planes12


def get_depth_from_plane_param(ranges, normal, offset, max_depth = cfg.dataset.MAX_DEPTH):
    """
    Params:
        normal: [B, 3, h, w]
        offsets: [B, 1, h, w]
    Return:
        Zs: [B, H, W]
    """
    # Z = o/ ((x - c_xd ) / f_xd * n_x + (y - c_yd )/f_yd * n_y + n_z)
    d = torch.sum(ranges * normal, dim=1) # n^T * kinvq
    d[d==0] = 1e-4
    Zs = offset.squeeze() / d 
    if max_depth > 0:
        return torch.clip(Zs, 0, max_depth)
    return Zs 

def get_3Dpts_from_depth(depth, ranges):
    """
    Params:
        depth: B, 1, h, w
        ranges: B, 3, N
    Return:
        XYZs: B, 3, N
    """
    b, c, h, w = depth.shape
    Zs = depth.reshape(b, c, -1) 

    return ranges * Zs


def get_sampling_grid(depth, transform12, intrinsics, height, width):
    """
    given set of sampled plane hypotheses, pixel location, and camera parameters -> return warped location for each hypothesis
    Params:
        depth: shape [B, C, H, W]
        transform12 (matrix): camera parameters converting view1 -> world -> view2 [4 x 4]
        intrinsics (matrix): scaled depth intrinsic params shape [B x 4 x 4]

    Return:
        u, v: shape [B, H, W] # warped x_indices and y_indices separate
        outprojection_mask: [B, H, W]
    """
    fx = intrinsics[:,0,0]
    fy = intrinsics[:,1,1]
    mx = intrinsics[:,0,2]
    my = intrinsics[:,1,2]
    ranges = get_imgtocamcoord_batch(intrinsics, height, width) # B, 3, N
    XYZs = get_3Dpts_from_depth(depth, ranges) 
    B, _, N = XYZs.shape

    # rotate to view image2
    ones = torch.ones(B,1,N, device=XYZs.device)
    pt = torch.cat((XYZs, ones), dim=1) # B, 4, N
    pt_tr = torch.bmm(transform12, pt) # B, 4, N
    Zs_tr = pt_tr[:, 2, :] # B, N

    # project back 2D
    pts = pt_tr[:, 0:3, :] / (Zs_tr.unsqueeze(1) + 1e-5)  # B, 3, N

    intrinsics = intrinsics[:,0:3,0:3]
    uvs = torch.bmm(intrinsics, pts) # B, 3, N
    us = uvs[:,0,:].reshape(B, height, width)
    vs = uvs[:,1,:].reshape(B, height, width)
    outprojection_mask = (us > width) | (us < 0) | (vs > height) | (vs < 0)
    return us, vs, outprojection_mask


def grid_sampler(us,vs,image):
    """
    Apply bilinear interpolation using projected_indices and feature maps
    Params:
        image: [B, C, H, W] 
        us, vs: [B, H, W]

    Return:
        warped: [B, C, H, W]

    """
    B, C, H, W = image.shape
    if B != us.shape[0]:
        image = image.repeat(us.shape[0], 1,1,1)
    # normalize values
    u = 2*us / (W-1)  - 1
    v = 2*vs / (H-1)  - 1
    # reshape
    uv = torch.stack((u,v), dim=-1)
    warped = F.grid_sample(image, uv, align_corners=True) 

    return warped

