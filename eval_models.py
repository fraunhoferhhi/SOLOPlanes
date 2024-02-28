from collections import defaultdict
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from collections import defaultdict
from functools import partial

from config.solopmv_cfg import config as cfg_solop
from datasets.scannet_single import ScannetSingle
from modules.pred_heads.plane_pred_head import PlanePredHead
from utils.util import *
from utils.plane_utils import *
from utils.eval import *
from modules.misc import multi_apply
import code
import matplotlib.cm as cm

def solop_eval_mAP(model, dataloader, fname, iou_thr=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        classes = np.arange(1,42)
        score_thresholds = np.arange(0.5,1,0.05)
        recall_dict = defaultdict(lambda: np.zeros(score_thresholds.shape))
        precision_dict = defaultdict(lambda: np.zeros(score_thresholds.shape))
        batch_w_class = defaultdict(lambda: np.zeros(score_thresholds.shape))
        for batch in tqdm(dataloader): 
            image = batch['image'].to(device)
            gt_masks = batch['inst_masks']
            inst_labels = batch['inst_semantic_labels']
            gt_masks = [torch.tensor(m, dtype=torch.float, device=device) for m in gt_masks]

            seg, planes = model.predict(image)
            for c in classes:
                for i, score in enumerate(score_thresholds):
                    seg_masks, soft_masks, cate_label, cate_scores = list(zip(*seg))
                    get_ious = partial(calc_iou, score_thr=score)
                    res = list(map(get_ious,  seg_masks, gt_masks, cate_label, cate_scores))
                    ious, pred_labels = list(zip(*res))
                    precisions, recalls, contains_class = multi_apply(get_pr_mAP, ious, pred_labels, inst_labels, class_id=c, iou_thr=iou_thr)
                    n_samples = torch.tensor(contains_class).sum()
                    if n_samples == 0:
                        continue
                    batch_w_class[c][i] += 1
                    pre = torch.tensor(precisions).sum() / n_samples 
                    rec = torch.tensor(recalls).sum() / n_samples
                    recall_dict[c][i] += rec.item()
                    precision_dict[c][i] += pre.item()

        recall_dict = {k: v / batch_w_class[k] for k,v in recall_dict.items()}
        precision_dict = {k: v / batch_w_class[k]  for k,v in precision_dict.items()}
        mAP = calc_mAP(precision_dict, recall_dict)
        print(f"mAP: {mAP}")
        np.savez(fname + "_pr-mAP.npz", precision_dict=precision_dict, recall_dict=recall_dict, mAP=mAP)

def solop_eval_seg(model, dataloader, fname, iou_thr=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        score_thresholds = np.arange(0.5,1,0.05)
        total_recall = np.zeros(score_thresholds.shape)
        total_precision = np.zeros(score_thresholds.shape)
        for batch in tqdm(dataloader): 
            image = batch['image'].to(device)
            gt_masks = batch['inst_masks']
            inst_labels = batch['inst_semantic_labels']
            gt_masks = [torch.tensor(m, dtype=torch.float, device=device) for m in gt_masks]

            seg, planes = model.predict(image)
            for i, score_thr in enumerate(score_thresholds):
                seg_masks, soft_masks, cate_label, cate_scores = list(zip(*seg))
                get_ious = partial(calc_iou, score_thr=score_thr)
                res = list(map(get_ious,  seg_masks, gt_masks, cate_label, cate_scores))
                ious, _  = list(zip(*res))
                precisions, recalls = multi_apply(get_precision_recall, ious, iou_thr=iou_thr)
                total_recall[i] += torch.tensor(recalls).mean().item()
                total_precision[i] += torch.tensor(precisions).mean().item()
        total_recall /= len(dataloader)
        total_precision /= len(dataloader)
        ap = calc_ap(total_precision, total_recall)
        np.savez(fname + "_pr-AP.npz", precisions=total_precision, recalls=total_recall, ap=ap)
        print(f"AP {ap}")

def solop_eval_depth(model, dataloader, fname, eval_mode='depth', mask_thr=0.55, use_pp_inst_planes = True):
    """per pixel recall using depth and normals for diff thresholds"""
    assert eval_mode in ["depth", "normal"], "choose valid eval mode"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_dict = defaultdict(float)
    with torch.no_grad():
        if eval_mode=="depth":
            thresholds = np.arange(0,1.01,0.02)
        else:
            thresholds = np.arange(0,0.26,0.01)
        model_recall = np.zeros(thresholds.shape)
        for batch in tqdm(dataloader): 
            image = batch['image'].to(device)
            depth = batch['depth'].to(device)
            gt_planes = batch['perpix_planes'].to(device)
            nonplanar_mask = batch['nonplanar_mask'].to(device)
            ranges = batch['kinv_xy1'].to(device)

            seg, planes = model.predict(image)
            if use_pp_inst_planes:
                filtered_segs = multi_apply(filter_masks_by_threshold, seg, score_thr=mask_thr) 
                filtered_segs = tuple(map(list, zip(*filtered_segs)))
                seg_masks, soft_masks, cate_label, cate_scores = multi_apply(sort_masks_by_density, filtered_segs)
                inst_planes, inst_perpix_planes = multi_apply(inst_plane_pooling, soft_masks, seg_masks, planes)
                planes = torch.stack(inst_perpix_planes)
            if eval_mode == 'depth':
                n, d = get_plane_params(planes)
                induced_depth = get_depth_from_plane_param(ranges, n, d)
                recalls = eval_induced_depth(induced_depth, depth, thresholds)
                metrics = calc_depth_metrics(induced_depth, depth)
                for k,v in metrics.items():
                    metric_dict[k] += v.item()
            else:
                recalls = eval_normal_sim(planes, gt_planes,thresholds, ~nonplanar_mask)
            model_recall += np.array(recalls)
        model_recall = model_recall / len(dataloader)
        for k,v in metric_dict.items():
            metric_dict[k] /= len(dataloader)
            metric_dict[k] = round(metric_dict[k], 3)
        np.savez(fname + "_depth-metrics.npz", recalls=model_recall, thresholds=thresholds)
        print(f"saved {fname + '_depth-metrics.npz'}")
        print(f"depth metrics {metric_dict}")


if __name__=="__main__":
    cfg = cfg_solop
    set_seed(cfg.seed)
    print(f"model {cfg.model.path}")
    model = cfg.model.init.load_from_checkpoint(cfg.model.path, cfg=cfg, mode='val')
    model.eval()
    data = ScannetSingle(cfg, mode='test') 
    print(f"len data {len(data)}")

    fname = cfg.version
    b_size = 8
    n_workers = 4

    dataloader = DataLoader(
            data,
            b_size,
            collate_fn=ScannetSingle.collate,
            num_workers=n_workers,
            pin_memory=True)
    solop_eval_depth(model, dataloader, fname, eval_mode='depth')
    solop_eval_mAP(model, dataloader, fname)
    solop_eval_seg(model, dataloader, fname)
