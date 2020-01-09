# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


import sys
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, '../pytorch-unet'))
from geometry_plot import draw3DPts
from geometry import gramian, kern_mat, rgb_to_hsv

import threading

from cvo_utils import PtSampleInGrid

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def my_collate_fn(batch):
    batch_new = {}
    for item in batch[0]:
        batch_new[item] = {}
        if "velo_gt" not in item:
            batch_new[item] = torch.stack([batchi[item] for batchi in batch], 0)
        else:
            batch_new[item] = [batchi[item].unsqueeze(0) for batchi in batch]
    return batch_new

class Trainer:
    def __init__(self, options):
        self.opt = options

        ## data loader
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset, 
                         "TUM": datasets.TUMRGBDDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        self.device = torch.device("cuda:0")

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, collate_fn=my_collate_fn)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, collate_fn=my_collate_fn)
        self.val_iter = iter(self.val_loader)

        ## geometric transformation related: 
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.geo_scale = 0.1
        self.num_sp = 10
        color_nb_sub = torch.rand((self.num_sp, 2), device=self.device, dtype=torch.float32)
        self.color_nb = torch.cat((torch.zeros((self.num_sp, 1), device=self.device, dtype=torch.float32), color_nb_sub), dim=1 )
        self.color_nnb = torch.tensor([1,0,0], device=self.device, dtype=torch.float32)
        # ## save current opts
        # self.save_opts()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        for batch_idx, inputs in enumerate(self.train_loader):

            for key, ipt in inputs.items():
                if "velo_gt" not in key:
                    inputs[key] = ipt.to(self.device)
                else:
                    inputs[key] = [ipt_i.to(self.device) for ipt_i in ipt]

            outputs = {}
            for frame_id in self.opt.frame_ids:
                for scale in self.opt.scales:
                    ## gen_grid_flat
                    self.get_grid_flat(frame_id, scale, inputs, outputs)

                    ## sample a point
                    ## find the near points
                    self.samp_find_neighbor(frame_id, scale, outputs )

                    ## visualize the near and far points
                    for ib in range(self.opt.batch_size):
                        pts_xyz = outputs[("flat_xyz", frame_id, scale, frame_id, True)][ib]
                        pts_clr = outputs[("flat_nb_color", frame_id, scale, frame_id, True)][ib]
                        draw3DPts(pts_xyz.detach(), color_1=pts_clr.detach() )

    def samp_find_neighbor(self, frame_id, scale, outputs):

        outputs[("flat_nb_color", frame_id, scale, frame_id, True)] = {}
        for ib in range(self.opt.batch_size):
            num_pt = outputs[("flat_xyz", frame_id, scale, frame_id, True)][ib].shape[-1]
            outputs[("flat_nb_color", frame_id, scale, frame_id, True)][ib] = self.color_nnb.unsqueeze(1).expand(1, -1, num_pt).contiguous()

            neighbors = torch.zeros((1, self.num_sp, num_pt), device=self.device )
            # sps = torch.randint(num_pt, (self.num_sp,), device=self.device)
            sps = np.random.randint(low = 0, high = num_pt, size = self.num_sp) 
            for isp in range(self.num_sp):
                # print("flat_xyz", outputs[("flat_xyz", frame_id, scale, frame_id, True)][ib].shape)
                # print("isp", isp)
                # print("sps[isp], ", sps[isp])
                ref_xyz = outputs[("flat_xyz", frame_id, scale, frame_id, True)][ib][..., sps[isp]]
                diff = outputs[("flat_xyz", frame_id, scale, frame_id, True)][ib] - ref_xyz.unsqueeze(2)
                close_pts = (diff.norm(dim=1) < self.geo_scale).squeeze()
                # print("close_pts", close_pts)
                # print("close_pts.shape", close_pts.shape)
                # print("ref_xyz.shape", ref_xyz.shape)
                # true_pts = close_pts.nonzero().squeeze()
                # print("true_pts", true_pts)
                print("# of close points:", close_pts.sum())
                
                color_isp = self.color_nb[isp]
                # print("flat_color", outputs[("flat_nb_color", frame_id, scale, frame_id, True)][ib].shape)

                outputs[("flat_nb_color", frame_id, scale, frame_id, True)][ib][:, :, close_pts] = color_isp.unsqueeze(1)
                # outputs[("flat_nb_color", frame_id, scale, frame_id, True)][ib][0, 1, true_pts] = color_isp[1]
                # outputs[("flat_nb_color", frame_id, scale, frame_id, True)][ib][0, 2, true_pts] = color_isp[2]


    def flat_from_grid(self, grid_valid, grid_info_dict):
        ### ZMH: grid_xyz, grid_uv -> grid_valid, flat_xyz, flat_uv
        flat_info_dict = {}
        for item in grid_info_dict:
            flat_info_dict[item] = {}

        for i in range(self.opt.batch_size):
            mask_i = grid_valid[i].view(-1)
            for item in grid_info_dict:
                info_i = grid_info_dict[item][i]
                info_i = info_i.view(info_i.shape[0], -1)
                info_i_sel = info_i[:, mask_i]
                flat_info_dict[item][i] = info_i_sel.unsqueeze(0) # ZMH: 1*C*N

        return flat_info_dict

    def get_grid_flat(self, frame_id, scale, inputs, outputs):
        #### Generate: [pts (B*2*N), pts_info (B*C*N), grid_source (B*C*H*W), grid_valid (B*1*H*W)] in self frame and host frame
        #### outputs[("pts", frame_id, scale, frame_cd, gt_or_not)]
        for gt_flag in [True]:
        # for gt_flag in [True, False]:
            if gt_flag: 
                cam_pts_grid = self.backproject_depth[scale](
                    inputs[("depth_gt_scale", frame_id, scale)], inputs[("inv_K", scale)], as_img=True)
                outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)] = cam_pts_grid[:,:3] # ZMH: B*3*H*W
                outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)] = inputs[("depth_mask_gt", frame_id, scale)]  # ZMH: B*1*H*W
            else:
                cam_pts_grid = self.backproject_depth[scale](
                    outputs[("depth_scale", frame_id, scale)], inputs[("inv_K", scale)], as_img=True)
                outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)] = cam_pts_grid[:,:3]
                outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)] = inputs[("depth_mask", frame_id, scale)] 

            outputs[("grid_hsv", frame_id, scale, frame_id, gt_flag)] = rgb_to_hsv(inputs[("color", frame_id, scale)], flat=False)
                
            grid_info_dict = {}
            grid_info_dict["xyz"] = outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)]
            grid_info_dict["uv"] = self.backproject_depth[scale].id_coords.unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1) # ZMH: B*2*H*W
            grid_info_dict["hsv"] = outputs[("grid_hsv", frame_id, scale, frame_id, gt_flag)]
            grid_valid = outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)]
            flat_info_dict = self.flat_from_grid(grid_valid, grid_info_dict)
            outputs[("flat_xyz", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["xyz"]
            outputs[("flat_uv", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["uv"]
            outputs[("flat_hsv", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["hsv"]

            