import os
import sys
import pickle
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from utils.util import *
from utils.plane_utils import *
import logging
import copy
import code
import pdb


class ScannetBase(Dataset):
    """
    Parent class for ScannetSingle & ScannetMultiview data from ScanNet dataset
    Sample dict incl. RGB image, camera pose, intrinsics, depth, plane seg. mask & parameters.
    Scaled output for multiview model
    """
    def __init__( self, cfg, mode: str = 'train'):
        self.base_path = cfg.BASE_PATH
        self.dataroot = cfg.dataset.datafolder
        self.use_subset = cfg.dataset.use_subset
        self.scene_image_indices = []
        self.img_H = cfg.dataset.input_size[0]
        self.img_W = cfg.dataset.input_size[1]
        self.default_size = cfg.dataset.original_size
        self.DEPTH_SHIFT = 1000  # divide depth by 1000 for mm -> m
        self.scene_intrinsics = {}
        self.kinv_xy1_dict = {}
        self.mode = mode
        self.indices = {'train': cfg.dataset.train_indices, 'val': cfg.dataset.val_indices, 'test': cfg.dataset.test_indices}
        self.n_subset = {'train': cfg.dataset.train_subset, 'val': cfg.dataset.val_subset}

    def __len__(self):
        return len(self.scene_image_indices)

    def _fill_sceneimg_indices(self, scene_id, num_images):
        pass

    def _get_imtocamcoord(self,intrinsics, height, width):
        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        urange = (x - intrinsics[0,2]) / intrinsics[0,0]
        vrange = (y - intrinsics[1,2]) / intrinsics[1,1]
        kinvxy1 = np.stack([urange, vrange, np.ones(urange.shape)], axis=0) # 3, h, w
        return kinvxy1

    def set_scene_dat(self, scene_id):
        intrinsics = np.zeros((4,4), dtype=np.float32)
        scene_path = self.dataroot + '/scans/' + scene_id
        if not os.path.exists(scene_path + '/' + 
                scene_id + '.txt') or not os.path.exists(scene_path 
                        + '/annotation/planes.npy') or not os.path.isdir(scene_path + "/sens"):
            return False
        if len(os.listdir(scene_path + "/sens")) == 0:
            return False

        with open(scene_path + '/' + scene_id + '.txt') as f:
            for line in f:
                split_line = line.strip().split(' ')
                if split_line[0] == 'numColorFrames':
                    num_images = int(split_line[2])
                if split_line[0] == 'fx_depth':
                    intrinsics[0,0] = split_line[-1]
                if split_line[0] == 'fy_depth':
                    intrinsics[1,1] = split_line[-1]
                if split_line[0] == "mx_depth":
                    intrinsics[0,2] = split_line[-1]
                if split_line[0] == "my_depth":
                    intrinsics[1,2] = split_line[-1]

        intrinsics[2,2] = intrinsics[3,3] = 1
        self.scene_intrinsics[scene_id] = intrinsics
        ranges = self._get_imtocamcoord(intrinsics, self.img_H, self.img_W)
        self.kinv_xy1_dict[scene_id] = ranges.astype(np.float32)
        return num_images


    def fill_sceneimg_indices_precalc(self, mode):
        idx_dict_path = self.indices[mode]
        if type(idx_dict_path) == str:
            """Fill scene values and scene_image_indices in form [scene_id, (img_idx0, img_idx1...)]"""
            with open(idx_dict_path, 'rb') as f:
                scene_dict_set = pickle.load(f)
                for k in scene_dict_set.keys():
                    self.set_scene_dat(k)
                    for v in scene_dict_set[k]:
                        self.scene_image_indices.append([k, v])
        else:
            with open(self.dataroot + '/Tasks/Benchmark/scannetv2_' + mode + '.txt') as f:
                for line in f:
                    scene_id = line.strip()
                    num_images = self.set_scene_dat(scene_id)
                    if num_images:
                        self.scene_image_indices += self._fill_sceneimg_indices(scene_id, num_images)


    def get_sample(self, scene_id, img_idx):
        """
        Base for __getitem__ 
        Parameters:
            scene_id, img_idx (int): tuple item from self.scene_image_indices  
        Returns:
            data_dict (dict):   image,
                                depth, 
                                cam_pose, 
                                intrinsics, 
                                planes_gt_image, 
                                nonplanar_mask,
                                planes,
                                plane_segmentation
        """
        cam_pose = get_camera_params(
            self.dataroot + '/scans/' + scene_id + '/sens/pose/' + str(img_idx) + '.txt')
        intrinsics = self.scene_intrinsics[scene_id] 
        scene_kinvxy1 = self.kinv_xy1_dict[scene_id]
        
        segmentation_path = self.dataroot +  '/scans/' + scene_id + '/annotation/segmentation/' + str(img_idx) + '.png'
        plane_segmentation = load_image(segmentation_path, dtype=np.int32 )
        image_path = self.dataroot + '/scans/' + scene_id + '/sens/color/' + str(img_idx) + '.jpg'
        image = cv2.imread(image_path)

        depth_path = self.dataroot + '/scans/' + scene_id + '/sens/depth/' + str(img_idx) + '.png'
        depth = load_image(self.dataroot + '/scans/' + scene_id + '/sens/depth/' + str(img_idx) + '.png', dtype=np.float32 )

        if any(x is None for x in [image, intrinsics, cam_pose, depth]):
            return None

        depth = depth / self.DEPTH_SHIFT
        image = cv2.resize(image, (depth.shape[1], depth.shape[0]))

        # get edge binary
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (5, 5))
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=11)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=11)
        sobel_combined = np.hypot(sobelx, sobely)
        sobel_combined /= sobel_combined.max()
        normalized_min1max2_grad_im = (sobel_combined + 1).astype(np.float32)

        preprocessed_path = self.dataroot + "/preprocessed/" + scene_id + "-" + str(img_idx) + ".pkl"
        if os.path.isfile(preprocessed_path):
            with open(preprocessed_path, 'rb') as f:
                data = pickle.load(f)
            plane_idx_segmentation = data['plane_seg']
            planes = data['perpix_planes']
            plane_info = data['plane_info']
        else:
            planes, plane_info = load_planes(self.dataroot +
                                             '/scans/' +
                                             scene_id +
                                             '/annotation/planes.npy', self.dataroot +
                                             '/scans/' +
                                             scene_id +
                                             '/annotation/plane_info.npy')

            plane_idx_segmentation, planes, plane_info = process_gt_planes(image, planes, plane_segmentation, plane_info, depth, cam_pose, intrinsics)

            if planes is None:
                return None

            if len(planes) == 0 or plane_idx_segmentation.max() < 0:
                return None

            # convert plane coordinates
            planes[:,-1] = -planes[:,-1]
            planes[:,[1,2]] = planes[:,[2,1]]

            # save in preprocessed folder
            os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
            with open(preprocessed_path, 'wb') as f:
                pickle.dump({'plane_seg': plane_idx_segmentation, 'perpix_planes': planes, 'plane_info': plane_info}, f)

        # create plane param targets
        nonplanar_mask = plane_idx_segmentation == -1

        plane_params_gt = planes[plane_idx_segmentation]
        plane_params_gt[nonplanar_mask] = -1

        instance_masks = []
        plane_params = []
        inst_label = []
        for i, plane in enumerate(planes):
            m = plane_idx_segmentation == i 
            if m.sum() < 1:
                continue
            instance_masks.append(m)
            plane_params.append(planes[i])
            inst_label.append(plane_info[i][0][1]) # nyu label in scannet-v2-labels.combined.tsv
        inst_masks = np.stack(instance_masks, axis=2)
        bboxes = extract_bboxes(inst_masks)

        resize_transforms_list = [T.ToTensor()]
        image_transforms_list = [T.ToTensor()] 
        if self.img_H != self.default_size[0]:
            resize_transforms_list.append(T.Resize((self.img_H, self.img_W), interpolation= T.InterpolationMode.NEAREST))
            image_transforms_list.append(T.Resize((self.img_H, self.img_W), antialias=True))
        nn_resize_transforms = T.Compose(resize_transforms_list)
        image_transforms = T.Compose(image_transforms_list)

        depth = nn_resize_transforms(depth)
        plane_params_gt = nn_resize_transforms(plane_params_gt)
        plane_idx_segmentation = nn_resize_transforms(plane_idx_segmentation)
        nonplanar_mask = nn_resize_transforms(nonplanar_mask)
        normalized_min1max2_grad_im = nn_resize_transforms(normalized_min1max2_grad_im)
        image = image_transforms(image)

        data_dict = {
            'image': image,
            'gradient_im': normalized_min1max2_grad_im,  # values between 1-2
            'depth': depth,
            'cam_pose': torch.from_numpy(cam_pose),
            'intrinsics': torch.from_numpy(intrinsics),
            'kinv_xy1': torch.from_numpy(scene_kinvxy1), 
            'perpix_planes': plane_params_gt,
            'plane_idx_segmentation': plane_idx_segmentation,
            'nonplanar_mask': nonplanar_mask,
            'inst_planes': torch.from_numpy(np.asarray(plane_params)),
            'inst_masks': np.asarray(instance_masks),
            'inst_bboxes': torch.from_numpy(bboxes),
            'inst_semantic_labels': torch.tensor(inst_label, dtype=torch.int8)
            }

        return data_dict 

    @staticmethod
    def _collate(batch):
        '''custom collate function'''
        collated_batch = {}
        for k in batch[0].keys():
            if k.startswith('inst'):
                collated_batch[k] = [sample[k] for sample in batch]
            else:
                collated_batch[k] = torch.stack([sample[k] for sample in batch])
        return collated_batch


