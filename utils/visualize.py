import os
import open3d as o3d
import torchvision.transforms as T
import numpy as np
import torch
import cv2
from scipy import ndimage
import time
from matplotlib import pyplot as plt
from utils.util import *

def convert_tensor_image(image):
    image = image.squeeze()*255
    return image.permute(1, 2, 0).numpy().astype(np.uint8)

def drawPlaneImage(plane_seg):
    if not isinstance(plane_seg, np.ndarray):
        plane_seg = plane_seg.numpy()
    im = plane_seg - plane_seg.min()
    im = im / im.max()
    return (im * 255).astype(np.uint8)

def drawDepthImage(depth, maxDepth=10):
    depthImage = np.clip(depth / maxDepth * 255, 0, 255).numpy()
    return 255 - depthImage

def drawNormalImage(normal):
    normalImage = np.clip((normal + 1) / 2 * 255, 0, 255).numpy().astype(np.uint8)
    return normalImage

def map_loss_to_cmap(loss, cmap='jet'):
    lmax = loss.max()
    lmin = loss.min()
    loss = (loss - loss.min()) / (loss.max() - loss.min())
    cm = plt.get_cmap(cmap)
    loss_rgb = cm(loss.cpu())
    return loss_rgb

def show_result_ins(img, result, nyu40_labels,):
    if isinstance(img, str):
        img = cv2.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    seg_label, soft_mask, cate_label, score  = result
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cate_label.cpu().numpy()
    score = score.cpu().numpy()

    num_mask = seg_label.shape[0]

    np.random.seed(42)
    color_masks = [np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)]

    kernel = np.ones((3, 3), np.uint8)
    for idx in range(num_mask):
        cur_mask = seg_label[idx, :, :]
        cur_mask = cv2.resize(cur_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        edge_color = color_mask*0.7
        # get edges
        eroded = cv2.erode(cur_mask, kernel)
        edge_mask = cur_mask - eroded
        
        cur_mask_bool = cur_mask.astype(bool)
        edge_mask_bool = edge_mask.astype(bool)
        img_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.5 + color_mask * 0.5
        img_show[edge_mask_bool] = img_show[edge_mask_bool] * 0.5 + edge_color * 0.5

        cur_cate = cate_label[idx]
        label_text = nyu40_labels[cur_cate]
        cur_score = score[idx]

        label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(img_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))  
    return img_show

def visualize_images(imgs:list, rows:int=1, cols:int=0, titles:list=None, row_titles:list=None, single_channel_cmap="plasma", grayscale_indices = [], tight=True):
    """
    Params:
        imgs: list 
    """
    if cols==0:
        cols = len(imgs) // rows
    fig, ax = plt.subplots(rows, cols)
    for i in range(rows):
        if row_titles is not None:
            ax[i, 0].set_title(row_titles[i])
        for j in range(cols):
            flat_idx = i*(cols-1) + i + j
            idx = (i,j) if rows > 1 else j
            if flat_idx in grayscale_indices:
                ax[idx].imshow(imgs[flat_idx]) if len(imgs[flat_idx].shape)==3 else ax[idx].imshow(imgs[flat_idx], cmap='gray', vmin=np.min(imgs[flat_idx]), vmax=np.max(imgs[flat_idx]))
            else:
                ax[idx].imshow(imgs[flat_idx]) if len(imgs[flat_idx].shape)==3 else ax[idx].imshow(imgs[flat_idx], cmap=single_channel_cmap, vmin=np.min(imgs[flat_idx]), vmax=np.max(imgs[flat_idx]))
            ax[idx].axis('off')
            if titles:
                ax[idx].set_title(titles[flat_idx])
    if tight:
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def vis_pointclouds_windows_worldview(pcd_list1, pcd_list2):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='model 1', width=960, height=540, left=0, top=0)
    for pcd in pcd_list1:
        vis.add_geometry(pcd)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='model 2', width=960, height=540, left=960, top=0)
    for pcd in pcd_list2:
        vis2.add_geometry(pcd)

    while True:
        for pcd in pcd_list1:
            vis.update_geometry(pcd)
        if not vis.poll_events():
            break
        vis.update_renderer()

        for pcd in pcd_list2:
            vis2.update_geometry(pcd)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()

def vis_pointclouds_windows(xyz_points, rgb_points, target_xyz=None, target_rgb=None):
    """
    compare 2 point clouds in separate windows
    params
        xyz_points:list len 2
        rgb_points:list len 2
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=0)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz_points[0])  
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_points[0])
    vis.add_geometry(pcd_o3d)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=0)
    pcd_o3d1 = o3d.geometry.PointCloud()
    pcd_o3d1.points = o3d.utility.Vector3dVector(xyz_points[1])  
    pcd_o3d1.colors = o3d.utility.Vector3dVector(rgb_points[1])
    vis2.add_geometry(pcd_o3d1)
    if target_xyz is not None:
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_xyz)  
        if target_rgb is not None:
            target.colors = o3d.utility.Vector3dVector(target_rgb)
        else:
            target.paint_uniform_color(np.array([0,1,1], dtype=np.float64))
        vis.add_geometry(target)
        vis2.add_geometry(target)
    while True:
        vis.update_geometry(pcd_o3d)
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis2.update_geometry(pcd_o3d1)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()

def vis_pointclouds_animation(xyz_points:list, rgb_points:list, target_xyz=None, target_rgb=None):
    """
    view pointcloud updates compared to uniform painted target, displayed in rgb at end
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_o3d = o3d.geometry.PointCloud()
    if target_xyz is not None:
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_xyz) 
        target.paint_uniform_color(np.array([0,1,1], dtype=np.float64))
    for i in range(len(xyz_points) + 1):
        if i==0:
            pcd_o3d.points = o3d.utility.Vector3dVector(xyz_points[i])  
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_points[i])
            vis.add_geometry(pcd_o3d)
            if target_xyz is not None:
                vis.add_geometry(target)
            ctr = vis.get_view_control()
            ctr.set_front(np.array([ -0.26, -0.34, -0.90], dtype=np.float64))
            ctr.set_lookat(np.array([ 0.034, 0.05, 1.60], dtype=np.float64))
            ctr.set_up(np.array([ 0.168, -0.935, 0.311], dtype=np.float64))
            time.sleep(1)
        elif i == len(xyz_points):
            vis.remove_geometry(pcd_o3d) 
            if target_rgb is not None:
                target.colors = o3d.utility.Vector3dVector(target_rgb)
                vis.add_geometry(target)
            ctr = vis.get_view_control()
            ctr.set_front(np.array([ -0.26, -0.34, -0.90], dtype=np.float64))
            ctr.set_lookat(np.array([ 0.034, 0.05, 1.60], dtype=np.float64))
            ctr.set_up(np.array([ 0.168, -0.935, 0.311], dtype=np.float64))
            time.sleep(1)
        else:
            pcd_o3d.points = o3d.utility.Vector3dVector(xyz_points[i])  
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_points[i])
            vis.update_geometry(pcd_o3d)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(2)

def get_pointcloud_worldview(xyz_points:list, rgb_points:list=None, target_xyz:list=None):
    """
    visualize sequence of point clouds (projected to world view)
    """
    pcd_list = []
    for i in range(len(xyz_points)):
        pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz_points[i])  # set pcd_np as the point cloud points
        if rgb_points is not None:
            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_points[i])
        pcd_list.append(pcd_o3d)
    if target_xyz is not None:
        for t in target_xyz:
            target = o3d.geometry.PointCloud() 
            target.points = o3d.utility.Vector3dVector(t) 
            target.paint_uniform_color(np.array([0,1,1], dtype=np.float64))
            pcd_list.append(target)
    return pcd_list

def get_pointcloud(xyz_points, rgb_points=None, target_xyz = None):
    """
    visualize single pointcloud (with rotation animation option)
    """
    def rotate_view(vis):
        vis.get_render_option().load_from_json(config.BASE_PATH + '/render_view.json')
        ctr = vis.get_view_control()
        ctr.rotate(3.0, 0.0)
        return False

    pcd_list = []
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz_points)  # set pcd_np as the point cloud points
    if rgb_points is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_points)
    else:
        pcd_o3d.paint_uniform_color(np.array([1,0,1], dtype=np.float64))
    if target_xyz is not None:
        target = o3d.geometry.PointCloud()  # create point cloud object
        target.points = o3d.utility.Vector3dVector(target_xyz)  # set pcd_np as the point cloud points
        target.paint_uniform_color(np.array([0,1,1], dtype=np.float64))
        pcd_list.append(target)
    pcd_list.append(pcd_o3d)
    return pcd_list

def vis_ptcloud(pcd_list, rotate=False, set_view = True):
    if rotate:
        o3d.visualization.draw_geometries_with_animation_callback(pcd_list, rotate_view)
    else:
        if set_view:
            o3d.visualization.draw_geometries(pcd_list, front = [ 0.05, 0.047, -0.99],
			lookat = [ 0.03, 0.051, 1.603],
			up = [ 0.181, -0.98, -0.037 ],
			zoom = 0.5)
        else:
            o3d.visualization.draw_geometries(pcd_list)
