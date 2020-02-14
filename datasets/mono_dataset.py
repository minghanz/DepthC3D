# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

import skimage
from skimage.morphology import binary_dilation, binary_closing

from kitti_utils import project_lidar_to_img, flip_lidar

from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch._six import int_classes as _int_classes

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'): ## ZMH: '.jpg' originally
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        if not self.load_depth:
            raise ValueError("Depth not available!")    ## add by Minghan. Make sure depth is always available

        # ### for dilation operation to generate mask for valid pixels
        self.dilate_struct = {}
        for i in range(self.num_scales):
            scale_d = max(35 - 8*i, 1) # 5
            self.dilate_struct[i] = np.ones((scale_d, scale_d))

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

        # for k in list(inputs):
        #     if "depth_gt_scale" in k and k[2] == -1:
        #         n, im, i = k
        #         frame = inputs[k]
        #         for i in range(self.num_scales):
        #             new_w = self.width // (2 ** i)
        #             new_h = self.height // (2 ** i)
        #             inputs[(n, im, i)] = skimage.transform.resize(frame, (new_h, new_w), order=0,anti_aliasing=False )
        
        # for k in list(inputs):
        #     if "depth_gt_scale" in k:
        #         n, im, i = k
        #         frame = inputs[k]
        #         frame = np.expand_dims(frame, 0)
        #         inputs[k] = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        ## ZMH: commented by me. Do not use the K written in script above. 
        # # adjusting intrinsics to match each scale in the pyramid
        # for scale in range(self.num_scales):
        #     K = self.K.copy()

        #     K[0, :] *= self.width // (2 ** scale)
        #     K[1, :] *= self.height // (2 ** scale)

        #     inv_K = np.linalg.pinv(K)

        #     inputs[("K", scale)] = torch.from_numpy(K)
        #     inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        # ZMH: the input color images are in range [0,1]

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            self.get_depth_related(folder, frame_index, side, do_flip, inputs)

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_depth_related(self, folder, frame_index, side, do_flip, inputs):
        raise NotImplementedError

class SamplerForConcat(Sampler):
    """For concated dataset, so that every sampled mini-batch are from the same sub-dataset (same size for convenient batching)
    Directly initialize from dataset instead of Sampler
    We need three sub samplers, and each time pop out batch size number of indices from one single subset. 
    """

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
            
        ### initialize one sampler for each sub dataset
        self.sub_sizes = [len(subset) for subset in self.data_source.datasets]

        sub_cumsum = self.data_source.cumulative_sizes.copy()
        sub_cumsum.insert(0, 0)
        sub_sampler_indices = [list(range(sub_cumsum[i], sub_cumsum[i+1]) ) for i in range(len(self.sub_sizes)) ]
        self.sub_samplers = [ SubsetRandomSampler(sub_idxs) for sub_idxs in sub_sampler_indices]

        self.sub_idxs = []
        for i in range(len(self.sub_sizes)):
            idx = [i] * self.sub_sizes[i]
            self.sub_idxs.extend(idx)
        assert len(self.sub_idxs) == len(self.data_source), "The number of samples are the the same as the sum of each subset"
        
        self.num_samples = len(self.sub_idxs)

    def __iter__(self): ## The dataloader creates the iterator object at the beginning of for loop
        sub_idx_idxs = torch.randperm(self.num_samples).tolist()
        list_of_sub_iters = [iter(sub_sampler) for sub_sampler in self.sub_samplers ]

        end_reached = [False for _ in self.sub_samplers]
        for idx_idx in sub_idx_idxs:
            if all(end_reached):
                break
            sub_idx = self.sub_idxs[idx_idx]
            batch = []
            while True:
                try:
                    idx = next(list_of_sub_iters[sub_idx])
                except StopIteration:
                    end_reached[sub_idx] = True
                    break
                else:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        break
            if len(batch) > 0 and not self.drop_last:
                yield batch


    def __len__(self):
        if self.drop_last:
            return sum( sub_size//self.batch_size for sub_size in self.sub_sizes )
        else:
            return sum( (sub_size+self.batch_size-1) // self.batch_size for sub_size in self.sub_sizes )