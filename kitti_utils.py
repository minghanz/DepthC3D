from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter

import torch

import PIL.Image as pil

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def lidar_pose_from_cam_pose(T_c, T_cl):
    T_l = torch.matmul(T_cl.inverse(), torch.matmul(T_c, T_cl) )
    return T_l

## ZMH: This is not used
def flip_K(P_rect_norm):
    """ZMH: We consider the lrflip of an image as the result of changing the camera intrinsic parameters.
    """
    P_flip = P_rect_norm.copy()
    P_flip[0,0] = - P_rect_norm[0,0]
    P_flip[0,2] = 1 - P_rect_norm[0,2]

    return P_flip

def flip_lidar(velo_rect, P_rect_norm):
    """ZMH: When image is flipped, the lidar points should also be flipped. 
    But the optical axis may not be at the center of image, 
    which means the action of flipping an image may not correspond to simply flipping the lateral coordinate. 
    Therefore we need P_rect (intrinsics of the rectified image) as an input
    """
    fx = P_rect_norm[0,0]
    cx = P_rect_norm[0,2]
    ratio = (1-2*cx)/fx
    velo_flip = velo_rect.copy()
    velo_flip[:, 0] = -velo_rect[:, 0] + ratio * velo_rect[:, 2]
    # ZMH: after this, need to -velo_rect[:, 2] / fx(real scale) if we are to project this point cloud to image plane

    return velo_flip

def project_lidar_to_img(pcl_lidar, P_rect_norm, im_shape, vel_depth=False):
    """pcl_lidar: n*4, P_rect_norm: 4*4, im_shape: 2(rows, cols)
    """
    P_rect_K = np.identity(4)
    P_rect_K[0,:] = P_rect_norm[0,:] * float(im_shape[1])
    P_rect_K[1,:] = P_rect_norm[1,:] * float(im_shape[0])
    P_rect_K = P_rect_K[:3,:] # 3*4
    
    velo_pts_im = np.dot(P_rect_K, pcl_lidar.T).T ## ZMH: n*3 (actually only the first three dim is useful after here) [uz, vz, z]
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis] ## ZMH: divided by depth [u, v, z]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]
        ## ZMH: optional, encode the 3rd dimension with x value (forward) in lidar space, 
        ## otherwise it is the depth in camera space.

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    # ZMH: In matlab code provided by KITTI (http://www.cvlibs.net/datasets/kitti/raw_data.php?type=calibration, https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_raw_data.zip), 
    # the coordinate calculated after multiplication with projection matrix is directly used in plotting. We know that MATLAB use index starting from 1, 
    # which means the projection matrix projects the left-most point in view to u=1. Here in python the index starts from 0, therefore 
    # we should manually subtract one from the result after projection matrix, so that the coordinate still represents the same location in image. 
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]  ## ZMH: n*3 after removing points out of view

    # project to image
    depth = np.zeros((im_shape[:2]))    # ZMH: [height, width]
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2] # ZMH: depth[v, u] = z
    ## ZMH: x,y <-> col, row <-> dim1, dim0 in an array

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32) ## ZMH: [height, width]

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    # P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    ### ZMH: calc K for this image
    # P_rect_norm[0, :] = P_rect_norm[0, :] / float(im_shape[0])
    # P_rect_norm[1, :] = P_rect_norm[1, :] / float(im_shape[1])
    # P_rect_norm = np.vstack((P_rect_norm, np.array([0,0,0,1.0]) ))
    # print("P_rect_norm:", P_rect_norm)

    K_inv = np.linalg.inv(P_rect[:3,:3])
    Kt = P_rect[:3, 3:4]
    t = np.dot(K_inv, Kt)
    P_rect_t = np.identity(4)
    P_rect_t[:3, 3:4] = t # ZMH: 4*4

    P_velo2rect = np.dot(P_rect_t, np.dot(R_cam2rect, velo2cam))

    P_rect_K = np.copy(P_rect)
    P_rect_K[:3, 3] = 0 # ZMH: 3*4

    P_rect_norm = np.identity(4) # ZMH: 4*4
    P_rect_norm[0, :] = P_rect_K[0, :] / float(im_shape[1])
    P_rect_norm[1, :] = P_rect_K[1, :] / float(im_shape[0])

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    if vel_depth:
        depth_gt = pil.open(velo_filename)

        return depth_gt, P_rect_norm, im_shape
    else:
        velo = load_velodyne_points(velo_filename) ## ZMH: the x,y,z,1 in n*4
        velo = velo[velo[:, 0] >= 0, :]

        # project the points to the camera
        ### ZMH: do it in two steps
        velo_rect = np.dot(P_velo2rect, velo.T).T ## ZMH: the lidar points in the rectified cam frame
        
        # depth = project_lidar_to_img(velo_rect, P_rect_norm, im_shape)

        return velo_rect, P_rect_norm, im_shape ### ZMHL also return the extrinsic (not the same in different date folders)
