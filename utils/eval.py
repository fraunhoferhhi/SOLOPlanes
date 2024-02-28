import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch

def calc_iou(pred_mask, gt_mask, pred_labels, score, score_thr=0.5):
    """
    Params
        pred_mask: P, H, W
        gt_mask: M, H, W

    Return
        iou: P, M 

    """
    pred_mask = pred_mask[score > score_thr]
    pred_labels = pred_labels[score > score_thr]
    pred_mask = pred_mask.float()
    n_pred = pred_mask.shape[0]
    if n_pred == 0:
        return 0, 0
    masks_a = pred_mask.view(n_pred, -1)
    masks_b = gt_mask.view(gt_mask.shape[0], -1)

    intersection = torch.matmul(masks_a, masks_b.T)
    area_a = masks_a.sum(-1).unsqueeze(1)
    area_b = masks_b.sum(-1).unsqueeze(0)
    union = area_a + area_b - intersection
    iou = intersection / union
    return iou, pred_labels 

def get_precision_recall(iou_matrix, iou_thr=0.5):
    """
    Params
        iou_matrix: P, M
    Returns
        precision, recall: float
    """
    if type(iou_matrix) is int:
        return torch.ones(1), torch.zeros(1)
    n_pred, n_gt = iou_matrix.shape
    max_iou = iou_matrix.max(0).values
    tp = max_iou >= iou_thr
    tp = tp.sum()
    recall = tp / n_gt
    if n_pred == 0:
        precision = 1
    else:
        precision = tp / n_pred
    return precision, recall

def get_pr_mAP(iou_matrix,  pred_label, gt_label, class_id, iou_thr=0.5):
    """
    Params:
        iou_matrix: [P, M]
        class_id: int
    Returns:
        precision: float
        recall: float
        contains_inst: bool
    """
    gmsk = gt_label == class_id
    if gmsk.sum() == 0:
        return 0, 0, False
    if type(iou_matrix) is int:
        return 1, 0, True 
    pred_label += 1
    pmsk = pred_label == class_id
    if pmsk.sum() == 0:
        return 1, 0, True
    iou_for_class = iou_matrix[:, gmsk]
    iou_for_class = iou_for_class[pmsk, :]
    pre, rec = get_precision_recall(iou_for_class, iou_thr)
    return pre, rec, True 
    
def calc_mAP(precision_dict, recall_dict):
    ap = [calc_ap(pre, rec) for pre,rec in zip(precision_dict.values(), recall_dict.values())]
    mAP = np.array(ap).sum() / len(ap)
    return mAP

def calc_ap(pre, rec):
    sort_idx = np.argsort(rec)
    rec = rec[sort_idx]
    pre = pre[sort_idx]
    
    rec = np.insert(rec, 0,0)
    pre = np.insert(pre, 0,1)
    pre[1:] = np.maximum(pre[1:], pre[:-1])
    ap = (rec[1:] - rec[:-1]) * pre[1:]
    return ap.sum()
    

def calc_depth_metrics(induced_depth, gt_depth, min_depth = 1/1000, max_depth=40):
    """
    Params
        induced_depth: B, H, W
        gt_depth: B, 1, H, W
    """
    metric_dict = dict()
    gt_depth  = gt_depth.squeeze()
    valid_mask = gt_depth >= min_depth
    pred = induced_depth[valid_mask]
    target = gt_depth[valid_mask]

    pred[pred < min_depth] = min_depth
    pred[pred > max_depth] = max_depth
    # abs rel
    abs_rel_diff = torch.abs(pred - target) / target
    abs_rel = torch.mean(abs_rel_diff)
    metric_dict['abs_rel'] = abs_rel
    # sq rel
    sq_rel_diff = ((target - pred)**2) / target
    sq_rel = torch.mean(sq_rel_diff)
    metric_dict['sq_rel'] = sq_rel
    # rmse
    mse = torch.mean((target - pred) **2)
    rmse = torch.sqrt(mse)
    metric_dict['rmse'] = rmse
    # rmse log
    mse_log = torch.mean((torch.log(target + 1e-8) - torch.log(pred + 1e-8)) **2)
    rmse_log = torch.sqrt(mse_log)
    metric_dict['log_rmse'] = rmse_log

    log10 = torch.mean(torch.abs(torch.log10(pred + 1e-08) - torch.log10(target + 1e-08)))
    metric_dict['log10'] = log10

    # delta
    r1 = pred / target
    r2 = target / pred
    thresh = torch.max(r1, r2) 
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    metric_dict['delta1'] = a1
    metric_dict['delta2'] = a2
    metric_dict['delta3'] = a3
    return metric_dict

def eval_induced_depth(induced_depth, gt_depth, depth_thresholds, valid=None): 
    """
    Evaluate per-pixel recall using different depth thresholds
    Params
        induced_depth: B, H, W
        gt_depth: B, 1, H, W
        depth_thresholds: array
        valid_mask: B, 1, H, W
    Return
        recalls: list

    """
    recalls = []
    depth_mask = gt_depth != 0
    if valid:
        valid_mask = torch.logical_and(depth_mask, valid)
    else:
        valid_mask = depth_mask
    induced_depth = induced_depth.unsqueeze(1)
    diff = torch.abs(induced_depth[valid_mask] - gt_depth[valid_mask])
    total_pix = torch.sum(valid_mask)
    for thresh in depth_thresholds:
        tot = torch.sum(diff<thresh)
        rec = (tot/total_pix).item()
        recalls.append(rec)
    return recalls


def eval_normal_sim(predicted_planes, gt_planes, cos_sim_thresholds, valid_mask):
    """
    Evaluate per-pixel recall based on normal similarity threshold
    Params
        predicted_planes: B, C, H, W
        gt_planes: B, C, H, W
        cos_sim_thresholds: array
        valid_mask: B, 1, H, W
    """
    recalls = []
    # valid_mask = valid_mask.repeat_interleave(3,dim=1)
    predicted_planes = predicted_planes*valid_mask + ~valid_mask
    gt_planes = gt_planes*valid_mask + ~valid_mask

    cos_sim = F.cosine_similarity(predicted_planes, gt_planes)
    diff = (1 - ((cos_sim+1)/2))
    diff[~valid_mask.squeeze()] = 100
    total_pix = torch.sum(valid_mask)
    for thresh in cos_sim_thresholds:
        tot = torch.sum(diff < thresh)
        rec = (tot/total_pix).item()
        recalls.append(rec)
    
    return recalls

