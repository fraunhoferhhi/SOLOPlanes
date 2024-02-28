import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
import torchvision.transforms as T
import logging
from config import config

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_image(image_path, gray=False, dtype=None):
    """
    load segmentation, depth, or rgb image
    """
    try:
        image = cv2.imread(image_path, -1).astype(dtype) if dtype else cv2.imread(image_path, -1)
        if gray:
            img_float32 = np.float32(image)
            image = cv2.cvtColor(img_float32, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return None
    return image

def unnormalize_image(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    inv_transform = T.Normalize(mean=(-mean / std), std=(1 / std))
    return inv_transform(image) * 255

def filter_masks_by_threshold(result, score_thr):
        seg_label, soft_mask, cate_label, score  = result
        vis_inds = score > score_thr
        seg_label = seg_label[vis_inds]
        soft_mask = soft_mask[vis_inds]
        cate_label = cate_label[vis_inds]
        score = score[vis_inds]
        return seg_label, soft_mask, cate_label, score

def sort_masks_by_confidence(result):
        seg_label, soft_mask, cate_label, score  = result
        orders = torch.argsort(score, descending=True)
        seg_label = seg_label[orders]
        soft_mask = soft_mask[orders]
        cate_label = cate_label[orders]
        score = score[orders]
        return seg_label, soft_mask, cate_label, score

def sort_masks_by_density(result):
        seg_label, soft_mask, cate_label, score  = result
        num_mask = seg_label.shape[0]
        mask_density = torch.zeros(num_mask)
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            mask_density[idx] = cur_mask.sum()
        orders = torch.argsort(mask_density)
        orders = orders.flip(dims=[0])
        seg_label = seg_label[orders]
        soft_mask = soft_mask[orders]
        cate_label = cate_label[orders]
        score = score[orders]

        return seg_label, soft_mask, cate_label, score

def get_camera_params(intrinsic_extrinsic_path):
    try:
        with open(intrinsic_extrinsic_path) as f:
            pose = np.asarray([[float(n) for n in line.split(" ")]
                               for line in f], np.float32)
    except Exception as e:
        logging.error(e)
        return None
    return pose

def extract_bboxes(mask):
    """https://github.com/NVlabs/planercnn
    Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        ## Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            ## x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            ## No mask for this instance. Might happen due to
            ## resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def mask_out_image(image, mask):
    # RGB image and single channel mask
    mask = np.repeat(mask[:,:,np.newaxis], 3, axis=-1)
    masked_image = image * mask
    return masked_image.astype(np.uint8)

def save_model(save_path, epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)

def transform_xyz(xyz, M):
    '''
    xyz: N, 3
    M: 4, 4
    '''
    xyz1 = np.concatenate((xyz, np.ones((len(xyz), 1))), 1)
    res = np.matmul(M, xyz1.T)
    return res.T[:, :3]

def create_3D_pointcloud(image, depth, intrinsics, normals=None):
    """
    params
        image: H, W, C
        depth: H, W
        intrinsics: 4, 4
        normals: H, W, C
    """
    h, w, c = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    urange = (x - intrinsics[0,2]) / intrinsics[0,0]
    vrange = (y - intrinsics[1,2]) / intrinsics[1,1]
    xyz_points = np.stack([urange*depth, vrange*depth, depth], axis=-1).reshape(-1, 3) 
    rgb_points = image.reshape(-1, 3)
    valid_depth_ind = np.where(depth.flatten() > 0)[0]
    if normals is not None:
        return xyz_points[valid_depth_ind,:], rgb_points[valid_depth_ind, :], normals.reshape(-1,3)[valid_depth_ind,:]
    return xyz_points[valid_depth_ind,:], rgb_points[valid_depth_ind, :]
