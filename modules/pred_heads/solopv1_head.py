import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import cv2

from ..misc import multi_apply, matrix_nms
from scipy import ndimage
from skimage.transform import rescale
import code

def get_instance_param(soft_mask, perpix_planes):
    weighted_planes = soft_mask * perpix_planes
    inst_param = weighted_planes.reshape(3, -1).sum(1) / soft_mask.sum()
    return inst_param

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_loss(pred, target):
    pred = pred.contiguous().view(pred.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(pred * target, 1)
    b = torch.sum(pred * pred, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

class SOLOPv1Head(nn.Module):
    
    def __init__(self,
                 num_classes,  #41 nyu40
                 in_channels,  # 256 fpn outputs
                 seg_feat_channels=256,   #seg feature channels 
                 stacked_convs=4,        #solov2 light set 2
                 strides=(4, 8, 16, 32, 64),  # [8, 8, 16, 32, 32],
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.33,
                 use_same_feats=True,
                 num_grids=None,  #[40, 36, 24, 16, 12],
                 ins_out_channels=64,  #128
                 loss_ins_weight=1,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOPv1Head, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs  #2
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.scale_ranges = scale_ranges
        self.same_feats = use_same_feats
        if use_same_feats:
            print("using same features for kern & cate")
        else:
            print("using split lvl features for kern & cate")
        
        self.ins_loss_weight = loss_ins_weight  #3.0
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):          
            # Layer 0 plus position information, x,y channels, cat to convolution output
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(nn.Sequential(
                nn.Conv2d(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=norm_cfg is None),

                    nn.GroupNorm(num_channels=self.seg_feat_channels,
                    num_groups=32),
                    
                    nn.ReLU(inplace=True)
                    ))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=norm_cfg is None),

                    nn.GroupNorm(num_channels=self.seg_feat_channels,
                    num_groups=32),

                    nn.ReLU(inplace=True)
                    ))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

    def init_weights(self):
        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, 0, 0.01)
                        # nn.init.kaiming_uniform_(con.weight, mode='fan_out', nonlinearity='relu')
                        
        for m in self.kernel_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        nn.init.normal_(con.weight, 0, 0.01)
                        # nn.init.kaiming_uniform_(con.weight, mode='fan_out', nonlinearity='relu')
                        

        nn.init.normal_(self.solo_cate.weight, 0, 0.01)
        nn.init.normal_(self.solo_kernel.weight, 0, 0.01)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                                       list(range(len(self.seg_num_grids))),
                                                       eval=eval)
        return cate_pred, kernel_pred
    
    def split_feats(self, feats):
        if self.same_feats:
            return (feats[0],
                    feats[1],
                    feats[2],)
        return ((feats[0], feats[2]),
                (feats[1],feats[3]),
                (feats[2],feats[4]))

    def forward_single(self, x, idx, eval=False):
        if self.same_feats:
            ins_kernel_feat = x
        else:
            ins_kernel_feat = x[0]
            cate_feat = x[1]
        # ins branch, concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        x, y = torch.meshgrid(x_range, y_range, indexing='xy')
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
        
        # kernel branch
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear',align_corners=False)
        if self.same_feats:
            cate_feat = kernel_feat[:, :-2, :, :] # no coordinates for category pred
        else:
            cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear',align_corners=False)

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        # cate branch
        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred


    def loss(self,
             cate_preds,
             kernel_preds,
             ins_pred,
             plane_pred,
             gt_instplane_list,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             ):
         
        
        mask_feat_size = ins_pred.size()[-2:]
        ins_label_list, plane_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_instplane_list,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list, 
            mask_feat_size=mask_feat_size)

        # ins list 5 lvls, inst masks stacked across batch 
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]
        plane_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*plane_label_list)]

        # filter to match target
        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]

        # generate masks
        ins_pred_list = []
        for b_kernel_pred in kernel_preds: # for 5 lvls
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred): # for ea batch
                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...] # mask feature [128, 120, 160]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0) # [1, 128, 120, 160]
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1) # [M, 128, 1, 1]
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W) # [M, 120, 160]
                soft_mask = torch.sigmoid(cur_ins_pred)
                b_mask_pred.append(soft_mask)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0) # stack all predictions in batch
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        # dice loss
        l1 = nn.L1Loss()
        loss_ins = []
        for pred, target in zip(ins_pred_list, ins_labels): #, ins_plane_pred_list, plane_labels):
            if pred is None:
                continue
            loss_ins.append(dice_loss(pred, target))

        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [torch.cat([cate_labels_level_img.flatten() for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)]

        flatten_cate_labels = torch.cat(cate_labels) - 1

        cate_preds = [cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds]
        flatten_cate_preds = torch.cat(cate_preds)
        flatten_cate_preds = flatten_cate_preds[flatten_ins_ind_labels]

        flat_onehot_labels = F.one_hot(flatten_cate_labels[flatten_ins_ind_labels], num_classes=self.num_classes).type(torch.float32)
        loss_cate = sigmoid_focal_loss(flatten_cate_preds, flat_onehot_labels, alpha=0.75, reduction='mean') 

        return dict(loss_ins=loss_ins, loss_cate=loss_cate)

    def solov2_target_single(self,
                            gt_inst_planes_raw,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               mask_feat_size):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        plane_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            plane_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device) # indicates inst hit

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                plane_label_list.append(torch.zeros([0, 3], device=device))
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_inst_planes = gt_inst_planes_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_hs = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_ws = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            for seg_mask, inst_plane, gt_label, half_h, half_w in zip(gt_masks, gt_inst_planes, gt_labels, half_hs, half_ws):
                if seg_mask.sum() == 0:
                   continue
                # mass center
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4) # [480, 640]
                center_h, center_w = ndimage.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid)) # grid position instance falls in
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                # fill grid with gt label for ea. level
                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = rescale(seg_mask, 1/4)
                seg_mask_tensor = torch.tensor(seg_mask, device=device) # [120, 160]

                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask_tensor.shape[0], :seg_mask_tensor.shape[1]] = seg_mask_tensor
                        cur_plane_param = torch.zeros(inst_plane.shape, dtype = inst_plane.dtype, device=device)
                        cur_plane_param[...] = inst_plane
                        ins_label.append(cur_ins_label)
                        plane_label.append(cur_plane_param)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            ins_label = torch.stack(ins_label, 0)
            plane_label = torch.stack(plane_label, 0)

            ins_label_list.append(ins_label)
            plane_label_list.append(plane_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, plane_label_list, cate_label_list, ins_ind_label_list, grid_order_list


    def get_seg(self, cate_preds, kernel_preds, mask_feat, cfg, img_shape):
        num_levels = len(cate_preds)
        featmap_size = mask_feat.size()[-2:]
        n_imgs = cate_preds[0].shape[0] # 5 lvls, [B, S, S, n_class]

        result_list = []
        for img_id in range(n_imgs):
            cate_pred_list = [cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)]

            mask_feat_list = mask_feat[img_id, ...].unsqueeze(0)
            kernel_pred_list = [kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                                for i in range(num_levels)]

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, mask_feat_list, kernel_pred_list,
                                         featmap_size, img_shape, img_shape, cfg)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       mask_feats,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       cfg,
                       ):

        assert len(cate_preds) == len(kernel_preds)

        h, w = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # get highest scoring class
        cate_preds, cate_labels = torch.max(cate_preds, dim=1)
        inds = (cate_preds > cfg['score_thr'])
        cate_scores = cate_preds[inds] 
        if len(cate_scores) == 0:
            return None

        # cate_labels & kernel_preds
        inds = inds.nonzero().squeeze() 
        kernel_preds = kernel_preds[inds] # [M, 128]
        cate_labels = cate_labels[inds]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0) # [40, 36, 24, 16, 12] --> [1600, 2896, 3472, 3728, 3872]
        strides = kernel_preds.new_ones(size_trans[-1])

        # strides [8, 8, 16, 32, 32]
        n_stage = len(self.seg_num_grids) 
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds]

        # mask encoding.
        I, N = kernel_preds.shape # M, 128
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        mask_feats_orig = F.conv2d(mask_feats, kernel_preds, stride=1).squeeze(0)
        mask_feats = mask_feats_orig.sigmoid()
        # mask.
        seg_masks = mask_feats > cfg['mask_thr']
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        mask_feats = mask_feats[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (mask_feats * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores # cate scores weighted w/ mask certainty

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg['nms_pre']: # keep 500 w/ top scores
            sort_inds = sort_inds[:cfg['nms_pre']]
        seg_masks = seg_masks[sort_inds, :, :]
        mask_feats = mask_feats[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=cfg['kernel'],sigma=cfg['sigma'], sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg['update_thr']
        if keep.sum() == 0:
            return None
        mask_feats = mask_feats[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg['max_per_img']:
            sort_inds = sort_inds[:cfg['max_per_img']]
        mask_feats = mask_feats[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        mask_feats = F.interpolate(mask_feats.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear', align_corners=False)[:, :, :h, :w]
        seg_masks = F.interpolate(mask_feats,
                               size=ori_shape[:2],
                               mode='bilinear',
                               align_corners=False).squeeze(0)
        seg_masks = seg_masks > cfg['mask_thr']
        return seg_masks, mask_feats.squeeze(), cate_labels, cate_scores
