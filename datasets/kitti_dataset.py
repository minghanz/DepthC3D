# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map, flip_lidar, project_lidar_to_img, generate_depth_map_lyft
from .mono_dataset import MonoDataset

from skimage.morphology import binary_dilation, binary_closing
import torch
import cv2

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class TUMRGBDDataset(MonoDataset):
    """
    For loading TUM data
    """
    def __init__(self, *args, **kwargs):
        super(TUMRGBDDataset, self).__init__(*args, **kwargs)
        # self.K = np.array([[525.0, 0, 319.5, 0], 
        #                    [0, 525.0, 239.5, 0], 
        #                    [0, 0, 1, 0], 
        #                    [0, 0, 0, 1]], dtype=np.float32)
        self.K = np.array([[0.8203125, 0, 0.49921875, 0], 
                           [0, 1.09375, 0.4989583, 0], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (640, 480)
        # self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    
    def check_depth(self):
        ### TUM dataset split files are in the format of sequence_folder rgb_file_name depth_file_name l
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name, "depth",
            "{:010d}.png".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    
    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path, folder, "depth", f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 5000

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_image_path(self, folder, frame_index):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "rgb", f_str)
        return image_path

class VKITTIDataset(MonoDataset):
    """For VKITTI2 dataset"""
    def __init__(self, *args, **kwargs):
        super(VKITTIDataset, self).__init__(*args, **kwargs)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 0, "3": 1, "l": 0, "r": 1}

        self.K_ = np.array([[725.0087, 0, 620.5, 0], 
                            [0, 725.0087, 187, 0], 
                            [0, 0, 1, 0], 
                            [0, 0, 0, 1]])

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])
        side = line[2]

        velo_filename = self.get_depth_path(scene_name, frame_index, side)
        
        return os.path.isfile(velo_filename)

    def get_image_path(self, folder, frame_index, side):
        f_str = "rgb_{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "clone", "frames", "rgb", "Camera_{}".format(self.side_map[side]), f_str)
        return image_path

    def get_depth_path(self, folder, frame_index, side):
        f_str = "depth_{:05d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path, folder, "clone", "frames", "depth", "Camera_{}".format(self.side_map[side]), f_str)
        return depth_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_path = self.get_depth_path(folder, frame_index, side)

        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 100
        depth[depth>500] = 0
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_depth_related(self, folder, frame_index, side, do_flip, inputs):
        """
        inputs["depth_gt"]
        inputs[("depth_gt_scale", i, -1)]
        inputs[("depth_mask", i, j)]
        inputs[("depth_mask_gt", i, j)]
        """
        depth_gt = self.get_depth(folder, frame_index, side, do_flip)

        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        for scale in range(self.num_scales):
            K = self.K_.copy()
            K[0, :] = K[0, :] / float(self.full_res_shape[0])
            K[1, :] = K[1, :] / float(self.full_res_shape[1])
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K).to(dtype=torch.float32)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K).to(dtype=torch.float32)

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("depth_gt_scale", i, -1)] = self.get_depth(folder, frame_index, other_side, do_flip)
            else:
                inputs[("depth_gt_scale", i, -1)] = self.get_depth(folder, frame_index + i, side, do_flip)
            
            for j in range(self.num_scales):
                new_w = self.width // (2 ** j)
                new_h = self.height // (2 ** j)

                depth_gt = skimage.transform.resize(inputs[("depth_gt_scale", i, -1)], (new_h, new_w), order=0, preserve_range=True, mode='constant', anti_aliasing=False)
                mask = depth_gt > 0
                mask = np.expand_dims(mask, 0)
                inputs[("depth_mask", i, j)] = torch.from_numpy(mask)

                depth_gt = np.expand_dims(depth_gt, 0)
                inputs[("depth_gt_scale", i, j)] = torch.from_numpy(depth_gt.astype(np.float32))
                inputs[("depth_mask_gt", i, j)] = inputs[("depth_mask", i, j)]
            
            depth_gt = np.expand_dims(inputs[("depth_gt_scale", i, -1)], 0)
            inputs[("depth_gt_scale", i, -1)] = torch.from_numpy(depth_gt.astype(np.float32))


class LyftDataset(MonoDataset):
    """ For lyft dataset
    """
    def __init__(self, *args, **kwargs):
        super(LyftDataset, self).__init__(*args, **kwargs)

        if self.width == 512 and self.height == 224: #256: #self.width == 512 and self.height == 416: (not cropped)
            self.full_res_shape = (1224, 1024)
            self.crop_rows = (244, 244) # (365, 200) # (212, 200)
            self.crop_cols = (0, 0)
        elif self.width == 640 and self.height == 352:
            self.full_res_shape = (1920, 1080)
            self.crop_rows = (0, 0)
            self.crop_cols = (0, 0) ## TODO: implement cropping here
        else:
            raise ValueError("width {}, height {} not recognized.".format(self.width, self.height))
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.full_res_shape = (self.full_res_shape[0]-self.crop_cols[0]-self.crop_cols[1], self.full_res_shape[1]-self.crop_rows[0]-self.crop_rows[1] )

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    
    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        ##(left, upper, right, lower) cropping https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop 
        color = color.crop((self.crop_cols[0], self.crop_rows[0], color.size[0]-self.crop_cols[1], color.size[1]-self.crop_rows[1]))
        # print(color.size)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_{}".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        """Cropping will affect depth_gt and P_rect_norm (normalized intrinsic matrix)
        """
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne/{:010d}.bin".format(int(frame_index)))

        calib_filename = os.path.join(
            self.data_path,
            folder,
            "calib/{:010d}.txt".format(int(frame_index)))

        velo_rect, cam_intr = generate_depth_map_lyft(calib_filename, velo_filename, self.side_map[side])

        ### handle cropping
        cam_intr[0,2] -= self.crop_cols[0]
        cam_intr[1,2] -= self.crop_rows[0]

        P_rect_norm = np.identity(4)
        P_rect_norm[0,:] = cam_intr[0,:] / float(self.full_res_shape[0])
        P_rect_norm[1,:] = cam_intr[1,:] / float(self.full_res_shape[1])

        depth_gt = project_lidar_to_img(velo_rect, P_rect_norm, self.full_res_shape[::-1], lyft_mode=True)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt, velo_rect, P_rect_norm

    def get_depth_related(self, folder, frame_index, side, do_flip, inputs):
        """
        inputs["depth_gt"]
        inputs[("depth_gt_scale", i, -1)]
        inputs[("depth_mask", i, j)]
        inputs[("depth_mask_gt", i, j)]
        """

        depth_gt, _, K_ = self.get_depth(folder, frame_index, side, do_flip)

        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        ## ZMH: load intrinsic matrix K here
        for scale in range(self.num_scales):
            K = K_.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K).to(dtype=torch.float32)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K).to(dtype=torch.float32)

        ## ZMH: load image with network-compatible size
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("depth_gt_scale", i, -1)], _, _ = self.get_depth(folder, frame_index, other_side, do_flip)
            else:
                inputs[("depth_gt_scale", i, -1)], velo_rect, P_rect_norm = self.get_depth(folder, frame_index + i, side, do_flip)
                if do_flip:
                    inputs[("velo_gt", i)] = flip_lidar(velo_rect, P_rect_norm)
                    inputs[("velo_gt", i)] = torch.from_numpy(inputs[("velo_gt", i)].astype(np.float32))
                else:
                    inputs[("velo_gt", i)] = torch.from_numpy(velo_rect.astype(np.float32))
            
            for j in range(self.num_scales):
                new_w = self.width // (2 ** j)
                new_h = self.height // (2 ** j)

                depth_gt = project_lidar_to_img(velo_rect, P_rect_norm, np.array([new_h, new_w]))
                if do_flip:
                    depth_gt = np.fliplr(depth_gt)

                ### generate mask from the lidar points
                # mask = binary_dilation(depth_gt, self.dilate_struct[j])
                mask = depth_gt.copy()
                # mask[int(new_h/2):] = 1
                mask = binary_closing(mask, self.dilate_struct[j])
                mask = np.expand_dims(mask, 0)
                inputs[("depth_mask", i, j)] = torch.from_numpy(mask)

                depth_gt = np.expand_dims(depth_gt, 0)
                inputs[("depth_gt_scale", i, j)] = torch.from_numpy(depth_gt.astype(np.float32))
                inputs[("depth_mask_gt", i, j)] = inputs[("depth_gt_scale", i, j)] > 0
            
            depth_gt = np.expand_dims(inputs[("depth_gt_scale", i, -1)], 0)
            inputs[("depth_gt_scale", i, -1)] = torch.from_numpy(depth_gt.astype(np.float32))

class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        velo_rect, P_rect_norm, im_shape  = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = project_lidar_to_img(velo_rect, P_rect_norm, self.full_res_shape[::-1])

        ### ZMH: changed by me. 
        ### The shape is just a little bit different, not huge resizing, because the depth_gt before resizing have a little bit different sizes
        # depth_gt = skimage.transform.resize(
        #     depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant') # don't use this one
        # depth_gt = skimage.transform.resize(
        #     depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant', anti_aliasing=False)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)
            # velo_rect = flip_lidar(velo_rect, P_rect_norm) # ZMH: add by me

        return depth_gt, velo_rect, P_rect_norm

    def get_depth_related(self, folder, frame_index, side, do_flip, inputs):

        depth_gt, _, K_ = self.get_depth(folder, frame_index, side, do_flip)
        
        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        ## ZMH: load intrinsic matrix K here
        for scale in range(self.num_scales):
            K = K_.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K).to(dtype=torch.float32)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K).to(dtype=torch.float32)

        ## ZMH: load image with network-compatible size
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("depth_gt_scale", i, -1)], _, _ = self.get_depth(folder, frame_index, other_side, do_flip)
            else:
                inputs[("depth_gt_scale", i, -1)], velo_rect, P_rect_norm = self.get_depth(folder, frame_index + i, side, do_flip)
                if do_flip:
                    inputs[("velo_gt", i)] = flip_lidar(velo_rect, P_rect_norm)
                    inputs[("velo_gt", i)] = torch.from_numpy(inputs[("velo_gt", i)].astype(np.float32))
                else:
                    inputs[("velo_gt", i)] = torch.from_numpy(velo_rect.astype(np.float32))
            
            for j in range(self.num_scales):
                new_w = self.width // (2 ** j)
                new_h = self.height // (2 ** j)

                depth_gt = project_lidar_to_img(velo_rect, P_rect_norm, np.array([new_h, new_w]))
                if do_flip:
                    depth_gt = np.fliplr(depth_gt)

                ### generate mask from the lidar points
                # mask = binary_dilation(depth_gt, self.dilate_struct[j])
                mask = depth_gt.copy()
                mask[int(new_h/2):] = 1
                mask = binary_closing(mask, self.dilate_struct[j])
                mask = np.expand_dims(mask, 0)
                inputs[("depth_mask", i, j)] = torch.from_numpy(mask)

                # depth_gt = skimage.transform.resize(inputs[("depth_gt_scale", i, -1)], (new_h, new_w), order=0, anti_aliasing=False ) # mine, ok
                # depth_gt = skimage.transform.resize(inputs[("depth_gt_scale", i, -1)], (new_h, new_w), order=0, preserve_range=True, mode='constant') # from kitti_dataset.py (KITTIRAWDataset), not ok
                # depth_gt = skimage.transform.resize(inputs[("depth_gt_scale", i, -1)], (new_h, new_w), order=0, preserve_range=True, mode='constant', anti_aliasing=False) # combined, ok
                depth_gt = np.expand_dims(depth_gt, 0)
                inputs[("depth_gt_scale", i, j)] = torch.from_numpy(depth_gt.astype(np.float32))
                inputs[("depth_mask_gt", i, j)] = inputs[("depth_gt_scale", i, j)] > 0
            
            depth_gt = np.expand_dims(inputs[("depth_gt_scale", i, -1)], 0)
            inputs[("depth_gt_scale", i, -1)] = torch.from_numpy(depth_gt.astype(np.float32))


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        # depth_gt = pil.open(depth_path) # ZMH: moved into generate_depth_map
        depth_gt, P_rect_norm, im_shape  = generate_depth_map(calib_path, depth_path, self.side_map[side], vel_depth=True)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt, P_rect_norm

    def get_depth_related(self, folder, frame_index, side, do_flip, inputs):
        """
        inputs["depth_gt"]
        inputs[("depth_gt_scale", i, -1)]
        inputs[("depth_mask", i, j)]
        inputs[("depth_mask_gt", i, j)]
        """
        depth_gt, K_ = self.get_depth(folder, frame_index, side, do_flip)
        
        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32)) # input original scale

        ## ZMH: load intrinsic matrix K here
        for scale in range(self.num_scales):
            K = K_.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K).to(dtype=torch.float32)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K).to(dtype=torch.float32)

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("depth_gt_scale", i, -1)], _ = self.get_depth(folder, frame_index, other_side, do_flip)
            else:
                inputs[("depth_gt_scale", i, -1)], _ = self.get_depth(folder, frame_index + i, side, do_flip)
                

            for j in range(self.num_scales):
                new_w = self.width // (2 ** j)
                new_h = self.height // (2 ** j)
                
                depth_gt = skimage.transform.resize(inputs[("depth_gt_scale", i, -1)], (new_h, new_w), order=0, preserve_range=True, mode='constant', anti_aliasing=False)

                mask = depth_gt.copy()
                mask[int(new_h/2):] = 1
                mask = binary_closing(mask, self.dilate_struct[j])
                mask = np.expand_dims(mask, 0)
                inputs[("depth_mask", i, j)] = torch.from_numpy(mask)
                
                depth_gt = np.expand_dims(depth_gt, 0)
                inputs[("depth_gt_scale", i, j)] = torch.from_numpy(depth_gt.astype(np.float32))
                inputs[("depth_mask_gt", i, j)] = inputs[("depth_gt_scale", i, j)] > 0

            depth_gt = np.expand_dims(inputs[("depth_gt_scale", i, -1)], 0)
            inputs[("depth_gt_scale", i, -1)] = torch.from_numpy(depth_gt.astype(np.float32))