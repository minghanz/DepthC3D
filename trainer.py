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
# from geometry_plot import draw3DPts
from geometry import gramian, kern_mat, rgb_to_hsv, hsv_to_rgb


sys.path.append(os.path.join(script_path, '../UPSNet'))
from upsnet.models import *
from wrap_to_panoptic import to_panoptic, PanopVis

import threading

from cvo_utils import PtSampleInGrid, PtSampleInGridAngle, PtSampleInGridWithNormal, calc_normal, recall_grad

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings("ignore")

import math
from pcl_vis import visualize_pcl

# import objgraph ## this is for debugging memory leak, but turns out not providing much useful information

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
    def __init__(self, options, ups_arg, ups_cfg):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        torch_vs = (torch.__version__).split('.')
        self.torch_version = float(torch_vs[0]) + 0.1 * float(torch_vs[1])

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:{}".format(self.opt.cuda_n))

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        if self.opt.use_panoptic:
            self.ups_cfg = ups_cfg # is none if self.opt.use_panoptic is None# create models
            self.ups_arg = ups_arg
            self.panoptic_model = eval(self.ups_cfg.symbol)().to(device=self.device)

            # preparing
            curr_iter = self.ups_cfg.test.test_iteration
            if self.ups_arg.weight_path == '':
                self.panoptic_model.load_state_dict(torch.load(os.path.join(os.path.join(os.path.join(self.ups_cfg.output_path, os.path.basename(self.ups_arg.cfg).split('.')[0]),
                                        '_'.join(self.ups_cfg.dataset.image_set.split('+')), self.ups_cfg.model_prefix+str(curr_iter)+'.pth'))), resume=True)
            else:
                self.panoptic_model.load_state_dict(torch.load(self.ups_arg.weight_path), resume=True)

            self.panop_visualizer = PanopVis(num_cls=50)


        # from apex import amp
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset, 
                         "kitti_depth": datasets.KITTIDepthDataset, 
                         "TUM": datasets.TUMRGBDDataset} # ZMH: kitti_depth originally not shown as an option here
        self.dataset = datasets_dict[self.opt.dataset]
        self.dataset_val = datasets_dict[self.opt.dataset_val]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        # self.train_loader = DataLoader(
        #     train_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, collate_fn=my_collate_fn)

        val_dataset = self.dataset_val(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        # self.val_loader = DataLoader(
        #     val_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, collate_fn=my_collate_fn)
        self.val_iter = iter(self.val_loader)
        self.val_count = 0

        self.ctime = time.ctime()
        ## create the path to log files (opt, model, writer, pcd, ...)
        ## in order to easily switch all loggers on or off by setting the paths to None
        if self.opt.disable_log:
            self.path_model = None
            self.path_opt = None
            self.path_pcd = None
            self.writers = None
        else:
            self.path_model = os.path.join(self.log_path, "models" + "_"+self.ctime)
            self.path_opt = self.path_model
            self.path_pcd = os.path.join(self.log_path, "pcds_"+self.ctime)
            self.writers = {}
            for mode in ["train", "val", "val_set"]:
                # self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode + '_' + self.ctime))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        self.set_other_params_from_opt() # moved from get_innerp_from_grid_flat

    def set_other_params_from_opt(self):
        
        if self.opt.dense_flat_grid:
            self.dist_combos = [(0, 1, True, False), (0, 0, True, False), (0, -1, True, False)]
            if self.opt.sup_cvo_pose_lidar:
                self.dist_combos.append((0, 1, True, True))
                self.dist_combos.append((0, -1, True, True))
            if self.opt.align_preds:
                self.dist_combos.append( (0, 1, False, False) )
                self.dist_combos.append( (0, -1, False, False) )
                
            # self.dist_combos = [(0, 1, False, False), (0, -1, False, False)]
            # inp_combos = self.inp_combo_from_dist_combo(dist_combos)

            if self.opt.use_panoptic:
                self.feats_cross = ["xyz", "seman"]
                self.feats_self = ["xyz", "panop"]
            else:
                self.feats_cross = ["xyz", "hsv"]
                self.feats_self = ["xyz", "hsv"]
            
            # feats_needed = ["xyz", "hsv"]
            self.feats_ell = {}
            # self.ell_base = 0.05
            self.ell_base = self.opt.ell_geo
            # if self.opt.random_ell:
            #     self.feats_ell["xyz"] = np.abs(self.ell_base* np.random.normal()) + 0.02
            # else:
            #     self.feats_ell["xyz"] = self.ell_base
            self.feats_ell["hsv"] = 0.2
            self.feats_ell["panop"] = 0.2    # in Angle mode this is not needed
            self.feats_ell["seman"] = 0.2    # in Angle mode this is not needed


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

        self.train_flag = True

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

        self.train_flag = False

    def train(self):
        """Run the entire training pipeline
        """
        # with torch.autograd.set_detect_anomaly(True):
        # with torch.autograd.detect_anomaly():
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        if self.opt.val_set_only:
            assert self.opt.load_weights_folder_parent is not None, "load_weights_folder_parent not given."
            self.run_val_set_only()
        else:
            for self.epoch in range(self.opt.num_epochs):
                torch.manual_seed(self.epoch) ## Jan 17, solve the problem of different shuffling of mini-batches between using and not using cvo trials after epoch 1. 
                np.random.seed(self.epoch)
                self.run_epoch()
                if (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.save_model()
                self.val_set()
    
    def run_val_set_only(self):
        """
        Use pretrained weights to run val set and log error to tensorboard
        """
        model_folder_list = os.listdir(self.opt.load_weights_folder_parent)
        # file_name_digit = [int(file_names[i].split('.')[0]) for i in range(len(file_names))]
        # file_name_idx = sorted(range(len(file_name_digit)),key=file_name_digit.__getitem__)
        # file_names = [file_names[i] for i in file_name_idx]

        model_folders = []
        model_seq = []
        for item in model_folder_list:
            path = os.path.join(self.opt.load_weights_folder_parent, item)
            if os.path.isdir(path):
                model_folders.append(path)
                weight_num=float(item.split('_')[-1])
                model_seq.append(weight_num)
        
        model_folders_idx = sorted(range(len(model_seq)), key=model_seq.__getitem__)
        model_folders = [model_folders[i] for i in model_folders_idx]

        
        for model_folder in model_folders:
            self.opt.load_weights_folder = model_folder
            self.load_model()

            self.val_set()
            self.step += 1


    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for self.batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            # if self.batch_idx < 20000:
            #     self.geo_scale = 1
            # elif self.batch_idx < 40000:
            #     self.geo_scale = 0.5
            # else:
            #     self.geo_scale = 0.1
            self.geo_scale = 0.1
            self.show_range = self.batch_idx % 1000 == 0

            outputs, losses = self.process_batch(inputs)

            ## ZMH: commented and use effective_batch instead
            ## https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672
            # self.model_optimizer.zero_grad()
            # ## losses['loss_cvo/hsv_tog_xyz_ori_'] or 'loss_cos/hsv_tog_xyz_tog_'
            # losses["loss"].backward()
            # self.model_optimizer.step()

            if self.batch_idx > 0 and self.batch_idx % self.opt.iters_per_update == 0:
                    self.model_optimizer.step()
                    self.model_optimizer.zero_grad()
                    # print('optimizer update at', iter_overall)
            if self.opt.cvo_as_loss:
                # loss = losses["loss_cos/hsv_tog_xyz_tog_"] / self.opt.iters_per_update
                # loss = ( losses["loss_cvo/hsv_ori_xyz_ori__2"] + losses["loss_cvo/hsv_ori_xyz_ori__3"] )/2 / self.opt.iters_per_update
                # loss = ( losses["loss_inp/hsv_ori_xyz_ori__2"] + losses["loss_inp/hsv_ori_xyz_ori__3"] )/2 / self.opt.iters_per_update
                # loss = ( losses["loss_inp/hsv_ori_xyz_ori__0"] + losses["loss_inp/hsv_ori_xyz_ori__1"] + losses["loss_inp/hsv_ori_xyz_ori__2"] + losses["loss_inp/hsv_ori_xyz_ori__3"] ) / self.opt.iters_per_update
                loss = ( losses["loss_inp/xyz_ori__0"] + losses["loss_inp/xyz_ori__1"] + losses["loss_inp/xyz_ori__2"] + losses["loss_inp/xyz_ori__3"] ) / self.opt.iters_per_update
                # loss = ( losses["loss_cvo/hsv_ori_xyz_ori__0"] + losses["loss_cvo/hsv_ori_xyz_ori__1"] + losses["loss_cvo/hsv_ori_xyz_ori__2"] + losses["loss_cvo/hsv_ori_xyz_ori__3"] ) / self.opt.iters_per_update
            else:
                loss = losses["loss"] / self.opt.iters_per_update
            if self.opt.disp_in_loss:
                loss += 0.1 * (losses["loss_disp/0"]+ losses["loss_disp/1"] + losses["loss_disp/2"] + losses["loss_disp/3"]) / self.num_scales / self.opt.iters_per_update
            if self.opt.supervised_by_gt_depth:
                # loss += 0.1 * losses["loss_cos/sum"] / self.num_scales / self.opt.iters_per_update
                # loss += 1e-6 * losses["loss_inp/sum"] / self.num_scales / self.opt.iters_per_update
                loss += losses["loss_inp/sum"] / self.num_scales / self.opt.iters_per_update
            if self.opt.sup_cvo_pose_lidar and not self.opt.dense_flat_grid:
                loss += 0.1 * losses["loss_pose/cos_sum"] / self.num_scales / self.opt.iters_per_update
            loss.backward()


            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = self.batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(self.batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

            ### ZMH: prevent memory leakage: https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770 
            # del loss, losses, outputs, inputs

            ## ZMH: monitor GPU usage:
            if early_phase or late_phase:
                allo = torch.cuda.memory_allocated()/1024/1024
                cach = torch.cuda.memory_cached()/1024/1024
                max_allo = torch.cuda.max_memory_allocated()/1024/1024
                max_cach = torch.cuda.max_memory_cached()/1024/1024

                print("GPU memory allocated at the end of iter {}: cur: {:.1f}, {:.1f}; max: {:.1f}, {:.1f}".format(self.step-1, allo, cach, \
                                                                                                                max_allo, max_cach ))
                # if (self.step-1) % 100 == 0:
                if self.writers is not None:
                    self.writers["train"].add_scalar("Mem/allo", allo, self.step-1)
                    self.writers["train"].add_scalar("Mem/cach", cach, self.step-1)
                    
                    self.writers["train"].add_scalar("Mem/max_allo", max_allo, self.step-1)
                    self.writers["train"].add_scalar("Mem/max_cach", max_cach, self.step-1)
                    
                # torch.cuda.reset_peak_stats()
                torch.cuda.reset_max_memory_cached()
                torch.cuda.reset_max_memory_allocated()
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass

            # objgraph.show_most_common_types()
            # objgraph.show_growth()
            # new_ids = objgraph.get_new_ids()
            # objgraph.get_leaking_objects()

            # print(self.step, '---------')
        
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if "velo_gt" not in key:
                inputs[key] = ipt.to(self.device)
            else:
                inputs[key] = [ipt_i.to(self.device) for ipt_i in ipt]

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
            ### ZMH: outputs have disp image of different scales

            ## ZMH: switch
            outputs_others = None
            ## ZMH: predict depth for each image (other than the host image)
            features_others = {} # initialize a dict
            outputs_others = {}
            for i in self.opt.frame_ids:
                if i == 0:
                    continue
                features_others[i] = self.models["encoder"](inputs["color_aug", i, 0])
                outputs_others[i] = self.models["depth"](features_others[i] )

        if self.opt.use_panoptic:
            with torch.no_grad():
                for i in self.opt.frame_ids:
                    list_of_tuple_for_panop = to_panoptic(inputs["color", i, 0], self.ups_cfg)
                    list_of_panop_feature = []
                    list_of_seman_feature = []
                    for ib in range(self.opt.batch_size):
                        panop_feature, seman_feature = self.panoptic_model( *( list_of_tuple_for_panop[ib] ) ) # 1*Catogories*H*W
                        panop_feature = F.softmax(panop_feature, dim=1)
                        seman_feature = F.softmax(seman_feature, dim=1)
                        list_of_panop_feature.append(panop_feature)
                        list_of_seman_feature.append(seman_feature)
                    outputs[("panoptic", i, 0)] = list_of_panop_feature 
                    outputs[("semantic", i, 0)] = torch.cat(list_of_seman_feature, dim=0) # This may not be viable fpr panop because different images may have different number of instances
                    for scale in self.opt.scales:
                        if scale == 0:
                            continue
                        outputs[("semantic", i, scale)] = F.interpolate(outputs[("semantic", i, 0)], scale_factor=0.5**scale)
                        outputs[("panoptic", i, scale)] = []
                        for ib in range(self.opt.batch_size):
                            rescaled_panop = F.interpolate(outputs[("panoptic", i, 0)][ib], scale_factor=0.5**scale)
                            outputs[("panoptic", i, scale)].append(rescaled_panop)

                    # mode = "train" if self.train_flag else "val"
                    # save_path = os.path.join(self.log_path, mode + '_' + self.ctime )
                    # self.panop_visualizer.paint(inputs["color", i, 0], outputs["panoptic", i, 0], save_path=save_path, step=self.step )

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
            ## ZMH: process depth for each image
            if outputs_others is not None:
                for i in self.opt.frame_ids:
                    if i == 0:
                        continue
                    outputs_others[i]["predictive_mask"] = self.models["predictive_mask"](features_others[i])

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # self.generate_depths_pred(inputs, outputs, outputs_others)

        # self.generate_images_pred(inputs, outputs)
        self.generate_images_pred(inputs, outputs, outputs_others)
        losses = self.compute_losses(inputs, outputs, outputs_others)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
            self.val_count = 0

        self.val_count += 1

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses
            print("Step:", self.step, "; Val_count:", self.val_count, "Epoch:", self.epoch, "Batch_idx:", self.batch_idx)

        self.set_train()

    def val_set(self):
        self.set_eval()
        print("------------------")
        print("Evaluating on the whole validating set at Epoch {}...".format(self.epoch))
        losses_sum = {}
        val_start_time = time.time()
        with torch.no_grad():
            self.geo_scale = 0.1
            self.show_range = False
            for batch_idx, inputs in enumerate(self.val_loader):
                outputs, losses = self.process_batch(inputs)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                for l, v in losses.items():
                    if batch_idx == 0:
                        losses_sum[l] = torch.tensor(0, dtype=torch.float32, device=self.device )
                    if type(v) != type(losses_sum[l]):
                        v = torch.tensor(v, dtype=torch.float32, device=self.device )
                    losses_sum[l] += v

                if batch_idx % 200 == 0:
                    print("Passed {} mini-batches in {:.2f} secs.".format(batch_idx, time.time()-val_start_time) )

            val_end_time = time.time()
            print("Val time: {:.2f}".format(val_end_time - val_start_time) )
            print("Total # of mini-batches in val set:", batch_idx+1)
            for l, v in losses_sum.items():
                losses_sum[l] = v / (batch_idx+1)
                print("{}: {:.2f}".format(l, losses_sum[l].item() ) ) # use .item to transform the 0-dim tensor to a python number      
            
            self.log("val_set", inputs, outputs, losses_sum)

            ### ZMH: prevent GPU memory leakage: 
            del inputs, outputs, losses, losses_sum

        self.set_train()
        print("--------------------")

    ### ZMH: make it a function to be repeated for images other than index 0
    def from_disp_to_depth(self, disp, scale, force_multiscale=False):
        """ZMH: generate depth of original scale unless self.opt.v1_multiscale
        """
        ## ZMH: force_multiscale option added by me to adapt to cases where we want multiscale depth
        if self.opt.v1_multiscale or force_multiscale:
            source_scale = scale
        else:
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        return depth, source_scale

    def gen_pcl_gt(self, inputs, outputs, disp, scale, frame_id, T_inv=None):
        ### ZMH: for host frame, T_inv is None.
        ### ZMH: gt depth -> point cloud gt (for other frames, transform the point cloud to host frame)
        ### ZMH: Due to that ground truth depth image is not valid at every pixel, the length of pointcloud in the mini-batch is not consistent. 
        ### ZMH: Therefore samples are processed one by one in the mini-batch.
        cam_points_gt, masks = self.backproject_depth[scale](
            inputs[("depth_gt_scale", frame_id, scale)], inputs[("inv_K", scale)], separate=True)
        if T_inv is None:
            outputs[("xyz_gt", frame_id, scale)] = cam_points_gt
            rgb_gt = {}
            for ib in range(self.opt.batch_size):
                color = inputs[("color", frame_id, scale)][ib]
                color = color.view(1, 3, -1)
                color_sel = color[..., masks[ib]]
                rgb_gt[ib] = color_sel
            outputs[("rgb_gt", frame_id, scale)] = rgb_gt
        else:
            cam_points_other_gt_in_host = {}
            rgb_other_gt = {}
            for ib in range(self.opt.batch_size):
                T_inv_i = T_inv[ib:ib+1]
                cam_points_other_gt_in_host[ib] = torch.matmul(T_inv_i, cam_points_gt[ib] )
                color = inputs[("color", frame_id, scale)][ib]
                color = color.view(1, 3, -1)
                color_sel = color[..., masks[ib]]
                rgb_other_gt[ib] = color_sel
            outputs[("xyz_gt", frame_id, scale)] = cam_points_other_gt_in_host
            outputs[("rgb_gt", frame_id, scale)] = rgb_other_gt

        ### ZMH: disparity prediction -> depth prediction -> point cloud prediction (for other frames, transform the point cloud to host frame)
        depth_curscale, _ = self.from_disp_to_depth(disp, scale, force_multiscale=True)
        cam_points_curscale = self.backproject_depth[scale](
            depth_curscale, inputs[("inv_K", scale)])
        if T_inv is None:
            xyz_is = {}
            rgb_is = {}
            for ib in range(self.opt.batch_size):
                xyz_is[ib] = cam_points_curscale[ib:ib+1, ..., masks[ib]]
                rgb_is[ib] = inputs[("color", frame_id, scale)][ib].view(1,3,-1)[..., masks[ib]]
            # outputs[("xyz_in_host", 0, scale)] = cam_points_curscale
            outputs[("xyz_in_host", frame_id, scale)] = xyz_is
            outputs[("rgb_in_host", frame_id, scale)] = rgb_is
        else:
            outputs[("depth", frame_id, scale)] = depth_curscale
            ### ZMH: transform points in source frame to host frame
            cam_points_other_in_host = torch.matmul(T_inv, cam_points_curscale)
            ### ZMH: log the 3d points to output (points in source frame transformed to host frame)
            ### ZMH: to sample the points only at where gt are avaiable: 
            xyz_is = {}
            rgb_is = {}
            for ib in range(self.opt.batch_size):
                xyz_is[ib] = cam_points_other_in_host[ib:ib+1, ..., masks[ib]]
                rgb_is[ib] = inputs[("color", frame_id, scale)][ib].view(1,3,-1)[..., masks[ib]]
            # outputs[("xyz_in_host", frame_id, scale)] = cam_points_other_in_host
            outputs[("xyz_in_host", frame_id, scale)] = xyz_is
            outputs[("rgb_in_host", frame_id, scale)] = rgb_is

    def gen_pcl_wrap_host(self, inputs, outputs, scale):

        ### 1. gt from lidar
        cam_points_gt, masks = self.backproject_depth[scale](
            inputs[("depth_gt_scale", 0, scale)], inputs[("inv_K", scale)], separate=True)
        outputs[("xyz_gt", 0, scale)] = cam_points_gt
        rgb_gt = {}
        for ib in range(self.opt.batch_size):
            color = inputs[("color", 0, scale)][ib]
            color = color.view(1, 3, -1)
            color_sel = color[..., masks[ib]]
            rgb_gt[ib] = color_sel
        outputs[("rgb_gt", 0, scale)] = rgb_gt

        ### 2. host frame same sampling
        masks = inputs[("depth_mask", 0, scale)]
        masks = [masks[i].view(-1) for i in range(masks.shape[0]) ]

        cam_points_host = self.backproject_depth[scale](
            outputs[("depth_wrap", 0, scale)], inputs[("inv_K", scale)] )
        xyz_is = {}
        rgb_is = {}
        for ib in range(self.opt.batch_size):
            xyz_is[ib] = cam_points_host[ib:ib+1, ..., masks[ib]]
            rgb_is[ib] = inputs[("color", 0, scale)][ib].view(1,3,-1)[..., masks[ib]]
        outputs[("xyz_in_host", 0, scale)] = xyz_is
        outputs[("rgb_in_host", 0, scale)] = rgb_is
        return masks

    def gen_pcl_wrap_other(self, inputs, outputs, scale, frame_id, T_inv, masks):
        ### 3. host frame by wrapping from adjacent frame
        uv_wrap = outputs[("uv_wrap", frame_id, scale)].view(self.opt.batch_size, 2, -1)
        ones_ =  torch.ones((self.opt.batch_size, 1, uv_wrap.shape[2]), dtype=uv_wrap.dtype, device=uv_wrap.device)
        own_id_coords = torch.cat((uv_wrap, 
                ones_), dim=1) # B*3*N
        
        cam_points_wrap = self.backproject_depth[scale](
            outputs[("depth_wrap", frame_id, scale)], inputs[("inv_K", scale)], own_pix_coords=own_id_coords )
        cam_points_other_in_host = torch.matmul(T_inv, cam_points_wrap)
        xyz_is = {}
        rgb_is = {}
        for ib in range(self.opt.batch_size):
            xyz_is[ib] = cam_points_other_in_host[ib:ib+1, ..., masks[ib]]
            rgb_is[ib] = outputs[("color_wrap", frame_id, scale)][ib].view(1,3,-1)[..., masks[ib]]
        outputs[("xyz_in_host", frame_id, scale)] = xyz_is
        outputs[("rgb_in_host", frame_id, scale)] = rgb_is

        ### 4. generate gt for adjacent frames
        cam_points_gt, masks = self.backproject_depth[scale](
            inputs[("depth_gt_scale", frame_id, scale)], inputs[("inv_K", scale)], separate=True)
        # cam_points_gt_in_host = torch.matmul(T_inv, cam_points_gt)
        # outputs[("xyz_gt", frame_id, scale)] = cam_points_gt_in_host
        outputs[("xyz_gt", frame_id, scale)] = cam_points_gt
        rgb_gt = {}
        for ib in range(self.opt.batch_size):
            color = inputs[("color", frame_id, scale)][ib]
            color = color.view(1, 3, -1)
            color_sel = color[..., masks[ib]]
            rgb_gt[ib] = color_sel
        outputs[("rgb_gt", frame_id, scale)] = rgb_gt

    def flat_from_grid(self, grid_valid, grid_info_dict):
        ### ZMH: grid_xyz, grid_uv -> grid_valid, flat_xyz, flat_uv
        flat_info_dict = {}
        for item in grid_info_dict:
            flat_info_dict[item] = {}

        for i in range(self.opt.batch_size):
            mask_i = grid_valid[i].view(-1)
            for item in grid_info_dict:
                if isinstance(grid_info_dict[item], list):
                    info_i = grid_info_dict[item][i][0]
                else:
                    info_i = grid_info_dict[item][i]
                info_i = info_i.view(info_i.shape[0], -1)
                info_i_sel = info_i[:, mask_i]
                flat_info_dict[item][i] = info_i_sel.unsqueeze(0) # ZMH: 1*C*N

        return flat_info_dict
    
    def flat_from_grid_single_item(self, grid_valid, grid_info):
        ### ZMH: grid_xyz, grid_uv -> grid_valid, flat_xyz, flat_uv
        flat_info = {}

        for i in range(self.opt.batch_size):
            mask_i = grid_valid[i].view(-1)
            if isinstance(grid_info, list):
                info_i = grid_info[i][0]
            else:
                info_i = grid_info[i]
            info_i = info_i.view(info_i.shape[0], -1)
            info_i_sel = info_i[:, mask_i]
            flat_info[i] = info_i_sel.unsqueeze(0) # ZMH: 1*C*N

        return flat_info

    def get_grid_flat(self, frame_id, scale, inputs, outputs):
        #### Generate: [pts (B*2*N), pts_info (B*C*N), grid_source (B*C*H*W), grid_valid (B*1*H*W)] in self frame and host frame
        #### outputs[("pts", frame_id, scale, frame_cd, gt_or_not)]

        ## if need sampling in mask
        if self.opt.mask_samp_as_lidar:
            # n_pts = inputs[("depth_mask_gt", frame_id, scale)].sum()
            mask_idx = inputs[("depth_mask_gt", frame_id, scale)].nonzero() # depth_mask
            from_n_pts = mask_idx.shape[0]
            n_pts = from_n_pts * 0.5
            idx_sample = torch.randperm(from_n_pts)[:int(n_pts)]

            mask_idx_sample = mask_idx[idx_sample]
            mask_idx_sample = mask_idx_sample.split(1, dim=1)
            mask_sp = torch.zeros_like(inputs[("depth_mask_gt", frame_id, scale)]) # depth_mask
            mask_sp[mask_idx_sample] = True
            inputs[("depth_mask_gt_sp", frame_id, scale)] = mask_sp

            mask_idx = inputs[("depth_mask", frame_id, scale)].nonzero()
            from_n_pts = mask_idx.shape[0]
            idx_sample = torch.randperm(from_n_pts)[:int(n_pts)]

            mask_idx_sample = mask_idx[idx_sample]
            mask_idx_sample = mask_idx_sample.split(1, dim=1)
            mask_sp = torch.zeros_like(inputs[("depth_mask", frame_id, scale)])
            mask_sp[mask_idx_sample] = True
            inputs[("depth_mask_sp", frame_id, scale)] = mask_sp

        for gt_flag in [True, False]:
            if gt_flag: 
                cam_pts_grid = self.backproject_depth[scale](
                    inputs[("depth_gt_scale", frame_id, scale)], inputs[("inv_K", scale)], as_img=True)
                outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)] = cam_pts_grid[:,:3] # ZMH: B*3*H*W
                if self.opt.mask_samp_as_lidar:
                    outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)] = inputs[("depth_mask_gt_sp", frame_id, scale)] 
                else:
                    outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)] = inputs[("depth_mask_gt", frame_id, scale)]  # ZMH: B*1*H*W
            else:
                cam_pts_grid = self.backproject_depth[scale](
                    outputs[("depth_scale", frame_id, scale)], inputs[("inv_K", scale)], as_img=True)
                outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)] = cam_pts_grid[:,:3]
                if self.opt.mask_samp_as_lidar:
                    outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)] = inputs[("depth_mask_sp", frame_id, scale)]
                else:
                    outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)] = inputs[("depth_mask", frame_id, scale)] 
                
            outputs[("grid_hsv", frame_id, scale, frame_id, gt_flag)] = rgb_to_hsv(inputs[("color", frame_id, scale)], flat=False)
            if self.opt.use_panoptic:
                outputs[("grid_panop", frame_id, scale, frame_id, gt_flag)] = outputs[("panoptic", frame_id, scale)]
                outputs[("grid_seman", frame_id, scale, frame_id, gt_flag)] = outputs[("semantic", frame_id, scale)]

            grid_info_dict = {}
            grid_info_dict["xyz"] = outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)]
            grid_info_dict["uv"] = self.backproject_depth[scale].id_coords.unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1) # ZMH: B*2*H*W
            grid_info_dict["hsv"] = outputs[("grid_hsv", frame_id, scale, frame_id, gt_flag)]
            if self.opt.use_panoptic:
                grid_info_dict["panop"] = outputs[("grid_panop", frame_id, scale, frame_id, gt_flag)]
                grid_info_dict["seman"] = outputs[("grid_seman", frame_id, scale, frame_id, gt_flag)]

            if self.opt.use_normal:
                self.normal_from_depth(frame_id, scale, outputs, gt_flag)
                grid_info_dict["normal"] = outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)]
                grid_info_dict["nres"] = outputs[("grid_nres", frame_id, scale, frame_id, gt_flag)]

            grid_valid = outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)]
            flat_info_dict = self.flat_from_grid(grid_valid, grid_info_dict)
            outputs[("flat_xyz", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["xyz"]
            outputs[("flat_uv", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["uv"]
            outputs[("flat_hsv", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["hsv"]
            if self.opt.use_panoptic:
                outputs[("flat_panop", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["panop"]
                outputs[("flat_seman", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["seman"]  

            if self.opt.use_normal:
                outputs[("flat_normal", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["normal"]  
                outputs[("flat_nres", frame_id, scale, frame_id, gt_flag)] = flat_info_dict["nres"] 

            self.combos_needed = None

            if frame_id == 0:
                id_pairs = [(0,1), (0,-1)]
            else:
                id_pairs = [(frame_id, 0)]

            for id_pair in id_pairs:
                combo = (id_pair[0], id_pair[1], gt_flag)
                need_calc = self.combos_needed is None or combo in self.combos_needed
                if need_calc:
                    other_id = id_pair[0] if id_pair[0] != 0 else id_pair[1] # ZMH: the one that is not zero
                    wrap_id = 0 if frame_id != 0 else other_id # ZMH: the one that is not current frame_id
                    
                    T = outputs[("cam_T_cam", 0, other_id)] # T_x0
                    outputs[("flat_xyz", frame_id, scale, wrap_id, gt_flag)] = {}
                    outputs[("flat_uv", frame_id, scale, wrap_id, gt_flag)] = {}
                    outputs[("flat_hsv", frame_id, scale, wrap_id, gt_flag)] = {}
                    if self.opt.use_panoptic:
                        outputs[("flat_panop", frame_id, scale, wrap_id, gt_flag)] = {}
                        outputs[("flat_seman", frame_id, scale, wrap_id, gt_flag)] = {}
                    if self.opt.use_normal:
                        outputs[("flat_normal", frame_id, scale, wrap_id, gt_flag)] = {}
                        outputs[("flat_nres", frame_id, scale, wrap_id, gt_flag)] = {}
                    for ib in range(self.opt.batch_size):
                        Ti = T[ib:ib+1]
                        Ki = inputs[("K", scale)][ib:ib+1, :3, :3]
                        ones = torch.ones( (1, 1, flat_info_dict["xyz"][ib].shape[2]) , dtype=flat_info_dict["xyz"][ib].dtype, device=self.device)
                        cam_homo = torch.cat( [flat_info_dict["xyz"][ib], ones], dim=1)

                        if frame_id == 0:
                            cam_wrap = torch.matmul(Ti, cam_homo)[:, :3]     # px = T_x0 * p0         # ZMH: B*3*N
                        else:
                            Ti_inv = torch.inverse(Ti)
                            cam_wrap = torch.matmul(Ti_inv, cam_homo)[:, :3] # p0 = T_0x * px         # ZMH: B*3*N

                        outputs[("flat_xyz", frame_id, scale, wrap_id, gt_flag)][ib] = cam_wrap
                        cam_uv_wrap = torch.matmul(Ki, cam_wrap) # ZMH: B*3*N
                        pix_coords = cam_uv_wrap[:, :2, :] / (cam_uv_wrap[:, 2, :].unsqueeze(1) + 1e-7) - 1 # ZMH: B*2*N
                        outputs[("flat_uv", frame_id, scale, wrap_id, gt_flag)][ib] = pix_coords.detach()
                        outputs[("flat_hsv", frame_id, scale, wrap_id, gt_flag)][ib] = outputs[("flat_hsv", frame_id, scale, frame_id, gt_flag)][ib]
                        if self.opt.use_panoptic:
                            outputs[("flat_panop", frame_id, scale, wrap_id, gt_flag)][ib] = outputs[("flat_panop", frame_id, scale, frame_id, gt_flag)][ib]
                            outputs[("flat_seman", frame_id, scale, wrap_id, gt_flag)][ib] = outputs[("flat_seman", frame_id, scale, frame_id, gt_flag)][ib]

                        if self.opt.use_normal:
                            if frame_id == 0:
                                Ri = Ti[:, :3, :3]
                            else:
                                Ri = Ti_inv[:, :3, :3]
                            outputs[("flat_normal", frame_id, scale, wrap_id, gt_flag)][ib] = torch.matmul(Ri, outputs[("flat_normal", frame_id, scale, frame_id, gt_flag)][ib])
                            outputs[("flat_nres", frame_id, scale, wrap_id, gt_flag)][ib] = outputs[("flat_nres", frame_id, scale, frame_id, gt_flag)][ib]

    def get_grid_flat_normal(self, outputs, n_pts_dict):
        """ after concat_flat, before calc inner product
        """
        ## loop over all frame_id and scale
        for scale in self.opt.scales:
            for frame_id in self.opt.frame_ids:
                for gt_flag in [True, False]:
                    ## 1. flat_normal and flat_nres from concat flat_xyz (same frame)
                    self.normal_from_depth_v2(outputs, frame_id, scale, gt_flag)

                    ## 2. grid_normal and grid_nres from flat_normal and flat_nres (same frame)
                    outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)], \
                        outputs[("grid_nres", frame_id, scale, frame_id, gt_flag)] = \
                        self.grid_from_concated_flat(flat_uvb=outputs[("flat_uv", frame_id, scale, frame_id, gt_flag)], \
                                                    flat_info=outputs[("flat_normal", frame_id, scale, frame_id, gt_flag)], \
                                                    flat_nres=outputs[("flat_nres", frame_id, scale, frame_id, gt_flag)], \
                                                    grid_xyz_shape=outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)].shape)

                    outputs[("grid_normal_vis", frame_id, scale, frame_id, gt_flag)] = outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)] * 0.5 + 0.5

                    ## 3. flat_normal and flat_nres of cross frames
                    n_pts = {}
                    n_pts[0] = 0
                    for ib in range(self.opt.batch_size):
                        n_pts[ib+1] = n_pts[ib] + n_pts_dict[("flat_uv", frame_id, scale, frame_id, gt_flag)][ib] # flat_uv is not flattened

                    if frame_id == 0:
                        id_pairs = [(0,1), (0,-1)]
                    else:
                        id_pairs = [(frame_id, 0)]

                    for id_pair in id_pairs:
                        combo = (id_pair[0], id_pair[1], gt_flag)
                        need_calc = self.combos_needed is None or combo in self.combos_needed
                        if need_calc:
                            other_id = id_pair[0] if id_pair[0] != 0 else id_pair[1] # ZMH: the one that is not zero
                            wrap_id = 0 if frame_id != 0 else other_id # ZMH: the one that is not current frame_id

                            Ts = outputs[("cam_T_cam", 0, other_id)] # T_x0
                            flat_normal_wrap_bs = []
                            for ib in range(self.opt.batch_size):
                                flat_normal_b = outputs[("flat_normal", frame_id, scale, frame_id, gt_flag)][:, :, n_pts[ib]:n_pts[ib+1] ]
                                Ti = Ts[ib:ib+1]
                                if frame_id == 0:
                                    Ri = Ti[:, :3, :3]
                                else:
                                    Ti_inv = torch.inverse(Ti)
                                    Ri = Ti_inv[:, :3, :3]

                                flat_normal_wrap_bs.append( torch.matmul(Ri, flat_normal_b ) )

                            outputs[("flat_normal", frame_id, scale, wrap_id, gt_flag)] = torch.cat(flat_normal_wrap_bs, dim=2)
                            outputs[("flat_nres", frame_id, scale, wrap_id, gt_flag)] = outputs[("flat_nres", frame_id, scale, frame_id, gt_flag)]

    def save_pcd_grad(self, frame_id, scale, gt_flag, n_pts, xyz_grad_grid, grid_valid, xyz):
        xyz_grad_flat = self.flat_from_grid_single_item(grid_valid, xyz_grad_grid) # not concatenated
        self.save_pcd_from_concat_flat(frame_id, scale, gt_flag, n_pts, xyz_grad=xyz_grad_flat, xyz=xyz)

    def save_pcd(self, outputs, n_pts_dict):
        frame_id = 0
        for scale in self.opt.scales:
            for gt_flag in [True, False]:
                n_pts = {}
                n_pts[0] = 0
                for ib in range(self.opt.batch_size):
                    n_pts[ib+1] = n_pts[ib] + n_pts_dict[("flat_uv", frame_id, scale, frame_id, gt_flag)][ib] 
                
                self.save_pcd_from_concat_flat(frame_id, scale, gt_flag, n_pts, outputs=outputs)

                if outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)].requires_grad:
                    grid_valid = outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)]
                    xyz = outputs[("flat_xyz", frame_id, scale, frame_id, gt_flag)]

                    outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)].register_hook(lambda grad,frame_id1=frame_id,scale1=scale,gt_flag1=gt_flag,n_pts1=n_pts,grir_valid1=grid_valid,xyz1=xyz: \
                                                                                            self.save_pcd_grad( frame_id1, scale1, gt_flag1, n_pts1, grad, grir_valid1, xyz1 ) )


    def save_pcd_from_concat_flat(self, frame_id, scale, gt_flag, n_pts, outputs=None, xyz_grad=None, xyz=None):
        with torch.no_grad():
            if self.path_pcd is None:
                return
            if not os.path.exists(self.path_pcd):
                os.makedirs(self.path_pcd)

            if xyz_grad is None:
                assert outputs is not None and xyz is None
            else:
                assert outputs is None and xyz is not None

            for ib in range(self.opt.batch_size):
                filename = os.path.join(self.path_pcd, "{}_{}_{}_{}_{}_{}".format(self.step, frame_id, scale, gt_flag, ib, self.train_flag) )

                if xyz_grad is None:
                    xyz = outputs[("flat_xyz", frame_id, scale, frame_id, gt_flag)][:, :, n_pts[ib]:n_pts[ib+1] ]
                    color = hsv_to_rgb( outputs[("flat_hsv", frame_id, scale, frame_id, gt_flag)][:, :, n_pts[ib]:n_pts[ib+1] ], flat=True )
                    if self.opt.use_normal or self.opt.use_normal_v2:
                        normal = outputs[("flat_normal", frame_id, scale, frame_id, gt_flag)][:, :, n_pts[ib]:n_pts[ib+1] ]
                        visualize_pcl(xyz, rgb=color, normal=normal, filename=filename, single_batch=True)
                    else:
                        visualize_pcl(xyz, rgb=color, filename=filename, single_batch=True)
                else:
                    xyz_ib = xyz[:, :, n_pts[ib]:n_pts[ib+1] ]
                    xyz_grad_ib = xyz_grad[ib] # xyz_grad is flat but not concatenated. It is a list
                    xyz_grad_ib_norm = xyz_grad_ib.norm(dim=1, keepdim=True) # B*1*N
                    grad_norm_max = xyz_grad_ib_norm.max()
                    xyz_grad_ib_norm = xyz_grad_ib_norm / grad_norm_max * 255
                    xyz_grad_ib = F.normalize(xyz_grad_ib, dim=1)
                    visualize_pcl(xyz_ib, intensity=xyz_grad_ib_norm, normal=xyz_grad_ib, filename=filename, single_batch=True, tag='_grad')
                    with open("{}_grad_norm_max.txt".format(filename), 'w') as f:
                        f.write( str(float(grad_norm_max)) )

    def normal_from_depth_v2(self, outputs, frame_id, scale, gt_flag):
        pts = outputs[("flat_uv", frame_id, scale, frame_id, gt_flag)]
        grid_source = outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)]
        grid_valid = outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)]
        neighbor_range = int(5)
        ignore_ib = False
        min_dist_2 = 0.05
        with torch.no_grad():
            normal, res = calc_normal(pts, grid_source, grid_valid, neighbor_range, ignore_ib, min_dist_2)
        outputs[("flat_normal", frame_id, scale, frame_id, gt_flag)] = normal
        outputs[("flat_nres", frame_id, scale, frame_id, gt_flag)] = res
        # self.print_range(res, pre_msg="{} {} {}".format(frame_id, scale, gt_flag))
    
    def print_range(self, tensor, pre_msg=None):
        with torch.no_grad():
            max_val = torch.max(tensor)
            max_dummy = torch.ones_like(tensor)*max_val
            tensor_nonzero = torch.where(tensor==0, max_dummy, tensor)
            min_val = torch.min(tensor_nonzero)
            print(pre_msg, float(min_val), float(max_val) )
        
    def grid_from_concated_flat(self, flat_uvb, flat_info, flat_nres, grid_xyz_shape):
        '''flat_info and flat_uvb: 1*C*N(sum of the mini-batch) or 1*3*N
        flat_nres: 1*1*N
        '''
        ## TODO: How to deal with points with no normal?
        uvb_split = flat_uvb.to(dtype=torch.long).squeeze(0).transpose(0,1).split(1,dim=1) # a tuple of 3 elements of tensor N*1, only long/byte/bool tensors can be used as indices

        C_info = flat_info.shape[1]
        grid_info = torch.zeros((grid_xyz_shape[0], C_info, grid_xyz_shape[2], grid_xyz_shape[3]), dtype=flat_info.dtype, device=flat_info.device) # B*C*H*W
        flat_info_t = flat_info.squeeze(0).transpose(0,1).unsqueeze(1) # N*1*C
        grid_info[uvb_split[2], :, uvb_split[1], uvb_split[0]] = flat_info_t

        C_res = flat_nres.shape[1]
        grid_nres = torch.zeros((grid_xyz_shape[0], C_res, grid_xyz_shape[2], grid_xyz_shape[3]), dtype=flat_info.dtype, device=flat_info.device) # B*1*H*W
        flat_nres_t = flat_nres.squeeze(0).transpose(0,1).unsqueeze(1) # N*1*1
        grid_nres[uvb_split[2], :, uvb_split[1], uvb_split[0]] = flat_nres_t

        return grid_info, grid_nres

    def normal_from_depth(self, frame_id, scale, outputs, gt_flag):
        #### generate depth normal and confidence
        ### 1. generate patches of points and valid map
        self.halfw_normal = 2 # from the visualization, using 2 results in large residual in closer part of image
        self.kern_normal = (2*self.halfw_normal+1, 2*self.halfw_normal+1)
        if gt_flag:
            kern_dilat = 2
        else:
            kern_dilat = 1
        self.equi_dist = 0.05
        # self.ref_nres = 0.1
        # self.ref_nres_sqrt = math.sqrt(self.ref_nres)

        num_in_kern = self.kern_normal[0]*self.kern_normal[1]
        xyz_grid = outputs[("grid_xyz", frame_id, scale, frame_id, gt_flag)]
        xyz_patches = F.unfold(xyz_grid, self.kern_normal, padding=self.halfw_normal*kern_dilat, dilation=kern_dilat) # B*(C*(2*self.halfw_normal+1)*(2*self.halfw_normal+1))*(H*W)
        xyz_patches = xyz_patches.reshape(self.opt.batch_size, 3, num_in_kern, -1).transpose(1,3) # B*(H*W)*N*3

        if gt_flag:
            valid_grid = outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)].to(dtype=torch.float32)
        else:
            valid_grid = torch.ones_like(outputs[("grid_valid", frame_id, scale, frame_id, gt_flag)]).to(dtype=torch.float32)

        if True: #not gt_flag:
            valid_patches = F.unfold(valid_grid, self.kern_normal, padding=self.halfw_normal*kern_dilat, dilation=kern_dilat) # B*(1*(2*self.halfw_normal+1)*(2*self.halfw_normal+1))*(H*W)
            valid_patches = valid_patches.reshape(self.opt.batch_size, 1, num_in_kern, -1).transpose(1,3).to(dtype=torch.bool) # B*(H*W)*N*1
            # valid_patches = torch.ones([xyz_patches.shape[0], xyz_patches.shape[1], xyz_patches.shape[2], 1], dtype=torch.bool, device=self.device) ## TODO: should consider padding

            valid_n_pts = valid_patches.sum(dim=2).to(dtype=torch.float32) # B*(H*W)*1
            valid_patches_expand = valid_patches.expand_as(xyz_patches) # B*(H*W)*N*3

        ### weight
        xyz_center = xyz_patches[:,:,[(num_in_kern-1)/2]] # B*(H*W)*1*3
        dist = torch.norm(xyz_patches - xyz_center, dim=3) # B*(H*W)*N
        # min_dist = torch.ones_like(dist) * self.equi_dist
        # dist = torch.where(dist > min_dist, dist, min_dist)
        dist = torch.clamp(dist, min=self.equi_dist)
        # dist = dist / self.equi_dist
        W_flat = 1/dist.unsqueeze(-1)   # B*(H*W)*N*1
        # W_diag = torch.diag_embed(W_flat) # B*(H*W)*N*N # this will cost a lot of memory
        # W_flat = W_flat.unsqueeze(-1)   # B*(H*W)*N*1
        # W_flat_3 = W_flat.expand_as(xyz_patches)
        # W_flat = torch.ones_like(W_flat)

        W_flat_sqrt = torch.sqrt(W_flat)

        if True:
            zero_vecs = torch.zeros_like(W_flat)
            W_flat = torch.where(valid_patches, W_flat, zero_vecs)
            W_flat_sqrt = torch.where(valid_patches, W_flat_sqrt, zero_vecs)
            W_flat_3_sqrt = W_flat_sqrt.expand_as(xyz_patches)

        ### 2. mask to only keep valid points
        if True: #not gt_flag:
            zero_patches = torch.zeros_like(xyz_patches) # B*(H*W)*N*3
            xyz_patches = torch.where(valid_patches_expand, xyz_patches, zero_patches)

        ### 3. reshape as n_pt * c
        ### 4. calc ATA, check inverse condition, substitude invalid tiles to diagonals
        # xyz_patches_t = xyz_patches.transpose(2,3) # B*(H*W)*3*N

        xyz_patches_w = xyz_patches * W_flat_3_sqrt
        xyz_patches_t_w = xyz_patches_w.transpose(2,3)
        
        # ATA = torch.matmul(xyz_patches_t, xyz_patches) # B*(H*W)*3*3
        ATA = torch.matmul(xyz_patches_t_w, xyz_patches_w ) # B*(H*W)*3*3
        detATA = torch.det(ATA) # B*(H*W)
        inverse_condition = (detATA > 1e-5).unsqueeze(-1).unsqueeze(-1).expand_as(ATA) # B*(H*W)*3*3
        
        diag = torch.diag(torch.ones(3, device=self.device)).view(1,1,3,3).expand_as(ATA)
        ATA_valid = torch.where(inverse_condition, ATA, diag)

        ### 5. calc (ATA)^(-1)AT1 on valid tiles
        inv_ATA = torch.inverse(ATA_valid) # # B*(H*W)*3*3
        ones_b = torch.ones((xyz_patches.shape[0], xyz_patches.shape[1], xyz_patches.shape[2], 1), device=self.device) # B*(H*W)*N*1
        # normal = torch.matmul(inv_ATA, torch.matmul(xyz_patches_t, ones_b) ) # B*(H*W)*3*1
        normal = torch.matmul(inv_ATA, torch.matmul(xyz_patches_t_w, ones_b*W_flat_sqrt ) ) # B*(H*W)*3*1
        # normal_norm = torch.norm(normal, dim=2, keepdim=True)
        # print("normal_norm min:", float(normal_norm.min()), "max:", float(normal_norm.max()), "mean:", float(normal_norm.mean()), "median:", float(normal_norm.median()) )
        normed_normal = normal / (torch.norm(normal, dim=2, keepdim=True)+1e-6) # B*(H*W)*3*1

        ### 6. calc error of Aw-1
        target_b = torch.matmul(xyz_center, normed_normal) # B*(H*W)*1*1
        residual = (torch.matmul(xyz_patches, normed_normal) - target_b ) * W_flat  # essentially res/dist = sin angle
        # residual = torch.matmul(xyz_patches, normal) - ones_b # B*(H*W)*N*1
        
        if True: #not gt_flag:
            residual = torch.where(valid_patches, residual, zero_vecs)

        # W_sum = W_flat.sum(dim=2)
        # print("W_sum min:", float(W_sum.min()), "max:", float(W_sum.max()), "mean:", float(W_sum.mean()), "median:", float(W_sum.median()) )
        # res_sum = (residual.pow(2) * W_flat ).sum(dim=2)
        # print("res_sum min:", float(res_sum.min()), "max:", float(res_sum.max()), "mean:", float(res_sum.mean()), "median:", float(res_sum.median()) )
        # if res_sum.min() <0:
        #     raise ValueError("res_sum.min() <0!")
        
        ## rmsq error
        # res_rmsq = torch.sqrt( (residual.pow(2) * W_flat ).sum(dim=2) / (W_flat.sum(dim=2)+1e-6) ) # instead of divided by valid_n_pts # B*(H*W)*1 # the ratio might be too large
        ## sqrt of mean abs with clamp
        # res_rmsq = torch.sqrt( torch.clamp((residual.abs() * W_flat).sum(dim=2) / (W_flat.sum(dim=2)+1e-6), min=0.01*self.ref_nres) )  # use clamp to avoid nan in back prop
        ## sqrt of mean abs
        # res_rmsq = torch.sqrt( (residual.abs() * W_flat).sum(dim=2) / (W_flat.sum(dim=2)+1e-6) )  # instead of divided by valid_n_pts # B*(H*W)*1 # will result in nan in back prop
        # mean abs
        res_rmsq = (residual.abs() * W_flat).sum(dim=2) / (W_flat.sum(dim=2)+1e-6)  # later use b/(b/t+x) to restrict the range

        ### 7. going back to flat/grid
        outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)] = normed_normal.squeeze(-1).transpose(1,2).reshape( [self.opt.batch_size, 3, xyz_grid.shape[-2], xyz_grid.shape[-1]] ) # B*3*H*W
        outputs[("grid_nres", frame_id, scale, frame_id, gt_flag)] = res_rmsq.transpose(1,2).reshape( [self.opt.batch_size, 1, xyz_grid.shape[-2], xyz_grid.shape[-1]] ) # B*1*H*W
        
        # norm_vis = outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)][:, 0:2] # B*2*H*W
        # norm_S = norm_vis.norm(dim=1) # [0,1]
        # norm_H = ( torch.atan2(norm_vis[:,0], norm_vis[:,1]) / math.pi + 1)/2 # [0,1]
        # norm_V = ( outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)][:, 2] + 1 )/2
        # norm_HSV = torch.stack([norm_H, norm_S, norm_V], dim=1) # B*3*H*W
        # outputs[("grid_normal_vis", frame_id, scale, frame_id, gt_flag)] = hsv_to_rgb(norm_HSV, flat=False)

        outputs[("grid_normal_vis", frame_id, scale, frame_id, gt_flag)] = outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)] * 0.5 + 0.5
        # outputs[("grid_normal_vis", frame_id, scale, frame_id, gt_flag)] = outputs[("grid_normal", frame_id, scale, frame_id, gt_flag)]
        
        # print("normal_res min:", float(res_rmsq.min()), "max:", float(res_rmsq.max()), "mean:", float(res_rmsq.mean()), "median:", float(res_rmsq.median()) ) # 0~1 as a sin angle
        # print("normed_normal min:", float(normed_normal.min()), "max:", float(normed_normal.max()), "mean:", float(normed_normal.mean()), "median:", float(normed_normal.median()) )
        # print("valid_n_pts min:", float(valid_n_pts.min()), "max:", float(valid_n_pts.max()), "mean:", float(valid_n_pts.mean()), "median:", float(valid_n_pts.median()) )

    def concat_flat(self, outputs):
        """
        This concat does not create new items in the outputs dict
        """
        n_pts_dict = {}
        for item in outputs:
            if "flat_" in item[0]:
                to_cat = []
                # new_item = item
                # if "flat_uv" in item[0]: ## "flat_uv" needs to be processed to "flat_uvb" 
                #     new_item = list(item)
                #     new_item[0] = "flat_uvb"
                #     new_item = tuple(new_item)
                if "flat_uv" in item[0]:
                    n_pts_dict[item] = {}

                for ib in range(self.opt.batch_size):
                    if "flat_uv" in item[0]:                    ## Create "flat_uvb"
                        n_pts = outputs[item][ib].shape[-1]
                        frame_indicater = torch.ones((1,1,n_pts), dtype=outputs[item][ib].dtype, device=self.device) * ib
                        flat_iuv = torch.cat([outputs[item][ib], frame_indicater ], dim=1) # here requires_grad=False
                        to_cat.append(flat_iuv)
                        n_pts_dict[item][ib] = n_pts
                    elif "flat_panop" in item[0]:               ## Do not concat "flat_panop"
                        continue # Don't concatenate, because diff images may have diff # of channels # each item of the list has 1*C*N
                    else:                                       ## Others just simply concat
                        to_cat.append(outputs[item][ib])

                if "flat_panop" not in item[0]:
                    outputs[item] = torch.cat(to_cat, dim=2)
        
        return n_pts_dict
                    
    def inp_combo_from_dist_combo(self, dist_combos):
        inp_combos = []
        for combo in dist_combos:
            inp_combos.append(combo)
            id0, id1, gt0, gt1 = combo
            combo0 = (id0, id0, gt0, gt0)
            combo1 = (id1, id1, gt1, gt1)
            if combo0 not in inp_combos:
                inp_combos.append(combo0)
            if combo1 not in inp_combos:
                inp_combos.append(combo1)
        return inp_combos
    
    def inp_feat_combo_from_dist_combo(self, dist_combos):
        inp_feat_combos = set()
        for combo in dist_combos:
            id0, id1, gt0, gt1 = combo
            if id0 == id1:
                feats_needed = self.feats_self
            else:
                feats_needed = self.feats_cross
            for feat in feats_needed:
                combo01 = (id0, id1, gt0, gt1, feat)
                combo00 = (id0, id0, gt0, gt0, feat)
                combo11 = (id1, id1, gt1, gt1, feat)
                inp_feat_combos.add(combo01)
                inp_feat_combos.add(combo00)
                inp_feat_combos.add(combo11)
        # print("inp_feat_combos:", inp_feat_combos)
        return inp_feat_combos

    def get_innerp_from_grid_flat_dummy(self, outputs):
        dist_dict = {}
        cos_dict = {}
        inp_dict = {}
        for combo in self.dist_combos:
            dist_dict[combo] = {}
            cos_dict[combo] = {}
            inp_dict[combo] = {}
            for scale in self.opt.scales:
                dist_dict[combo][scale] = torch.tensor(1., device=self.device)
                cos_dict[combo][scale] = torch.tensor(1., device=self.device)
                inp_dict[combo][scale] = torch.tensor(1., device=self.device)
        
        return inp_dict, dist_dict, cos_dict
                

    def get_innerp_from_grid_flat(self, outputs, n_pts_dict):
        
        if self.opt.random_ell:
            self.feats_ell["xyz"] = np.abs(self.ell_base* np.random.normal()) + self.opt.ell_min
        else:
            self.feats_ell["xyz"] = self.ell_base

        inp_feat_combos = self.inp_feat_combo_from_dist_combo(self.dist_combos)
        
        # neighbor_range = int(2)
        neighbor_range = self.opt.neighbor_range
        inp_feat_dict = {}
        for combo in inp_feat_combos:
            inp_feat_dict[combo] = {}
            flat_idx, grid_idx, flat_gt, grid_gt, feat = combo      # using the frame of grid_idx
            for scale in self.opt.scales:
                # pts, pts_info, grid_source, grid_valid, neighbor_range, ell
                ell = float(self.feats_ell[feat])
                if feat == "panop":
                    inn_list = []
                    n_pts = {}
                    n_pts[0] = 0
                    for ib in range(self.opt.batch_size):
                        grid_valid = outputs[("grid_valid", grid_idx, scale, grid_idx, grid_gt)][ib:ib+1]

                        n_pts[ib+1] = n_pts[ib] + n_pts_dict[("flat_uv", flat_idx, scale, grid_idx, flat_gt)][ib] # flat_uv is not flattened
                        flat_uv = outputs[("flat_uv", flat_idx, scale, grid_idx, flat_gt)][:, :, n_pts[ib]:n_pts[ib+1] ]

                        flat_info = outputs[("flat_"+feat, flat_idx, scale, grid_idx, flat_gt)][ib]
                        grid_info = outputs[("grid_"+feat, grid_idx, scale, grid_idx, grid_gt)][ib]

                        # print("grid_valid.shape", grid_valid.shape)
                        # print("flat_uv.shape", flat_uv.shape)
                        # print("flat_info.shape", flat_info.shape)
                        # print("grid_info.shape", grid_info.shape)
                        inn_single = PtSampleInGridAngle.apply(flat_uv.contiguous(), flat_info.contiguous(), grid_info.contiguous(), grid_valid.contiguous(), neighbor_range, True) # PtSampleInGrid
                        inn_list.append(inn_single)
                        # print("inn_single.shape", inn_single.shape)
                    inp_feat_dict[combo][scale] = torch.cat(inn_list, dim=2) #1 * NN * N
                else:
                    grid_valid = outputs[("grid_valid", grid_idx, scale, grid_idx, grid_gt)]
                    flat_uv = outputs[("flat_uv", flat_idx, scale, grid_idx, flat_gt)]
                    flat_info = outputs[("flat_"+feat, flat_idx, scale, grid_idx, flat_gt)]
                    grid_info = outputs[("grid_"+feat, grid_idx, scale, grid_idx, grid_gt)]
                    if feat == "seman":
                        inp_feat_dict[combo][scale] = PtSampleInGridAngle.apply(flat_uv.contiguous(), flat_info.contiguous(), grid_info.contiguous(), grid_valid.contiguous(), neighbor_range)
                    elif feat == "xyz" and (self.opt.use_normal or self.opt.use_normal_v2):
                        flat_normal = outputs[("flat_normal", flat_idx, scale, grid_idx, flat_gt)].contiguous()
                        grid_normal = outputs[("grid_normal", grid_idx, scale, grid_idx, grid_gt)].contiguous()
                        flat_nres = outputs[("flat_nres", flat_idx, scale, grid_idx, flat_gt)].contiguous()
                        grid_nres = outputs[("grid_nres", grid_idx, scale, grid_idx, grid_gt)].contiguous()
                        inp_feat_dict[combo][scale] = PtSampleInGridWithNormal.apply(flat_uv.contiguous(), flat_info.contiguous(), grid_info.contiguous(), grid_valid.contiguous(), \
                            flat_normal, grid_normal, flat_nres, grid_nres, neighbor_range, ell, self.opt.res_mag_max, self.opt.res_mag_min, False, self.opt.norm_in_dist, self.opt.ell_basedist)
                        
                        try:
                            assert not (inp_feat_dict[combo][scale]==0).all(), "{}{} is all zero".format(combo, scale)
                        except:
                            print("{}{} is all zero".format(combo, scale))
                            print("flat_normal", float(flat_normal.min()), float(flat_normal.max()) )
                            print("grid_normal", float(grid_normal.min()), float(grid_normal.max()) )
                            print("flat_nres", float(flat_nres.min()), float(flat_nres.max()) )
                            print("grid_nres", float(grid_nres.min()), float(grid_nres.max()) )

                            print("flat_info", float(flat_info.min()), float(flat_info.max()) )
                            print("grid_info", float(grid_info.min()), float(grid_info.max()) )
                            print("grid_valid", float(grid_valid.min()), float(grid_valid.max()) )
                            print("flat_uv", float(flat_uv.min()), float(flat_uv.max()) )


                        if  inp_feat_dict[combo][scale].requires_grad:
                             inp_feat_dict[combo][scale].register_hook(lambda grad: recall_grad(" inp_feat_dict", grad) )
                    else:
                        inp_feat_dict[combo][scale] = PtSampleInGrid.apply(flat_uv.contiguous(), flat_info.contiguous(), grid_info.contiguous(), grid_valid.contiguous(), neighbor_range, ell, False, False, self.opt.ell_basedist)

                        ### print stats of distances
                        # if feat == "xyz" and flat_gt == False and grid_gt == False :
                            
                        #     zs = inp_feat_dict[combo][scale]==0
                        #     inp_feat_dict[combo][scale][zs] = 100
                        #     print(inp_feat_dict[combo][scale].min())
                            # stats[0] = float(inp_feat_dict[combo][scale].min())
                            # stats[1] = float(inp_feat_dict[combo][scale].max())
                            # stats[2] = float(inp_feat_dict[combo][scale].median())
                            # stats[3] = float(inp_feat_dict[combo][scale].mean())
                            # print("{} to {} pred stats:".format(flat_idx, grid_idx), stats[0], stats[1], stats[2], stats[3])

                            # max_dist = - math.log(float(inp_feat_dict[combo][scale].max())) * 2*ell*ell
                            # median_dist = - math.log(float(inp_feat_dict[combo][scale].median())) * 2*ell*ell
                            # mean_dist = - math.log(float(inp_feat_dict[combo][scale].mean())) * 2*ell*ell
                            # # min_dist = - math.log(float(inp_feat_dict[combo][scale].min())) * 2*ell*ell
                            # print("{} to {} pred stats:".format(flat_idx, grid_idx), max_dist, median_dist, mean_dist)

                    # print("inp_feat_dict[{}][{}].shape".format(combo, scale), inp_feat_dict[combo][scale].shape)
        
        inp_dict, dist_dict, cos_dict = self.get_dist_from_inp_grid_flat(self.dist_combos, inp_feat_dict)

        return inp_dict, dist_dict, cos_dict

    def get_dist_from_inp_grid_flat(self, dist_combos, inp_feat_dict):
        innerp_dict = {} # building blocks for the other three
        dist_dict = {}
        cos_dict = {}
        inp_dict = {}
        for combo in dist_combos:
            id0, id1, gt0, gt1 = combo
            combo0 = (id0, id0, gt0, gt0)
            combo1 = (id1, id1, gt1, gt1)
            if id0 == id1:
                feats = self.feats_self
                flag = "self"
            else:
                feats = self.feats_cross
                flag = "cross"
            tags = []
            tags.append((combo, flag))
            tags.append((combo0, flag))
            tags.append((combo1, flag))
            for tag in tags:
                if tag in innerp_dict:
                    continue
                innerp_dict[tag] = {}
                combo_, _ = tag
                for scale in self.opt.scales:
                    cat_feat = torch.cat([inp_feat_dict[(*combo_, feat)][scale] for feat in feats], dim=0)
                    innerp_dict[tag][scale] = torch.prod(cat_feat, dim=0).sum()

                    if innerp_dict[tag][scale] == 0:
                        print(tag, scale)
                        for feat in feats:
                            print( feat, "All zero?", (inp_feat_dict[(*combo_, feat)][scale]==0).all() )


            dist_dict[combo] = {}
            cos_dict[combo] = {}
            inp_dict[combo] = {}
            for scale in self.opt.scales:
                # inp_dict[combo][scale] = - innerp_dict[tags[0]][scale] # fixed Jan 6!
                inp_dict[combo][scale] = innerp_dict[tags[0]][scale] # Jan 16: no neg here with log!
                dist_dict[combo][scale] = innerp_dict[tags[1]][scale] + innerp_dict[tags[2]][scale] - 2 * innerp_dict[tags[0]][scale]
                cos_dict[combo][scale] = 1 - innerp_dict[tags[0]][scale] / torch.sqrt( innerp_dict[tags[1]][scale] * innerp_dict[tags[2]][scale] )
                # cos_dict[combo][scale] = 1 - innerp_dict[tags[0]][scale] / torch.sqrt( torch.max( innerp_dict[tags[1]][scale] * innerp_dict[tags[2]][scale], torch.zeros_like(innerp_dict[tags[2]][scale])+1e-7 ) )

        return inp_dict, dist_dict, cos_dict

    def reg_cvo_to_loss_dummy(self, losses, inp_dict, dist_dict, cos_dict):
        losses["loss_cvo/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        losses["loss_cos/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        losses["loss_inp/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)

    def reg_cvo_to_loss(self, losses, inp_dict, dist_dict, cos_dict):
        losses["loss_cvo/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        losses["loss_cos/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        losses["loss_inp/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        for scale in self.opt.scales:
            for frame_id in self.opt.frame_ids:
                # if frame_id == 0:
                #     continue
                combo_ = (0, frame_id, True, False)
                losses["loss_cvo/{}_s{}_f{}".format(True, scale, frame_id)] = dist_dict[combo_][scale]
                losses["loss_cos/{}_s{}_f{}".format(True, scale, frame_id)] = cos_dict[combo_][scale]
                # losses["loss_inp/{}_s{}_f{}".format(True, scale, frame_id)] = inp_dict[combo_][scale]
                losses["loss_inp/{}_s{}_f{}".format(True, scale, frame_id)] = - self.feats_ell["xyz"]*self.feats_ell["xyz"]*torch.log( inp_dict[combo_][scale] ) # Jan 16: use log!
                try:
                    assert not torch.isnan(losses["loss_inp/{}_s{}_f{}".format(True, scale, frame_id)]).any(), "{} {} nan detected".format(scale, frame_id)
                    assert not torch.isinf(losses["loss_inp/{}_s{}_f{}".format(True, scale, frame_id)]).any(), "{} {} inf detected".format(scale, frame_id)
                except AssertionError as error:
                    print(scale, frame_id)
                    print("inp:", inp_dict[combo_][scale])
                    print("log inp:", torch.log( inp_dict[combo_][scale] )  )

                losses["loss_cvo/sum"] += losses["loss_cvo/{}_s{}_f{}".format(True, scale, frame_id)]
                losses["loss_cos/sum"] += losses["loss_cos/{}_s{}_f{}".format(True, scale, frame_id)]
                losses["loss_inp/sum"] += losses["loss_inp/{}_s{}_f{}".format(True, scale, frame_id)]
                
        return

    def get_xyz_dense(self, frame_id, scale, inputs, outputs):
        cam_points_gt = self.backproject_depth[scale](
            inputs[("depth_gt_scale", frame_id, scale)], inputs[("inv_K", scale)], as_img=True)

        cam_points_pred = self.backproject_depth[scale](
            outputs[("depth_scale", frame_id, scale)], inputs[("inv_K", scale)], as_img=True)

        outputs[("xyz1_dense_gt", frame_id, scale)] = cam_points_gt
        outputs[("xyz1_dense_pred", frame_id, scale)] = cam_points_pred

    def get_xyz_rgb_pair(self, frame_id, scale, inputs, outputs, gt):
        if gt:
            xyz1 = outputs[("xyz1_dense_gt", frame_id, scale)]
            mask = inputs[("depth_mask_gt", frame_id, scale)]
        else:
            xyz1 = outputs[("xyz1_dense_pred", frame_id, scale)]
            mask = inputs[("depth_mask", frame_id, scale)]

        rgb = inputs[("color", frame_id, scale)]
        hsv = rgb_to_hsv(rgb, flat=False)
        # print("hsv shape", hsv.shape)

        return xyz1, hsv, mask
    
    def get_xyz_aligned(self, id_pair, xyz1, outputs):
        xyz_aligned = [None]*2

        if id_pair[0] == id_pair[1]:
            xyz_aligned[0] = xyz1[0][:,:3]
        elif id_pair[0] == 0 and id_pair[1] != 0:
            ## TODO: Here other modes of pose prediction is not included yet.
            T = outputs[("cam_T_cam", 0, id_pair[1])] # T_x0
            height_cur = xyz1[0].shape[2]
            width_cur = xyz1[0].shape[3]
            xyz1_flat = xyz1[0].view(self.opt.batch_size, 4, -1)
            xyz1_trans = torch.matmul(T, xyz1_flat)[:,:3]
            xyz_aligned[0] = xyz1_trans.view(self.opt.batch_size, 3, height_cur, width_cur)
        else:
            raise ValueError("id_pair [{}, {}] not recognized".format(id_pair[0], id_pair[1]) )

        xyz_aligned[1] = xyz1[1][:,:3]
        return xyz_aligned

    def create_moving_slice(self, i, half_h, half_w, xyz_pair_off, mask_pair_off, hsv_pair_off, xyz_pair, mask_pair, hsv_pair, height, width ):
        for j in range(-half_w, half_w):
            off_h0_start = max(0, -i)
            off_h0_end = min(0, -i)
            off_w0_start = max(0, -j)
            off_w0_end = min(0, -j)
            off_h1_start = -off_h0_end
            off_h1_end = -off_h0_start
            off_w1_start = -off_w0_end
            off_w1_end = -off_w0_start
            

            # idx = (i + half_h) * (2 * half_w + 1) + j + half_w
            idx = j + half_w
            xyz_pair_off[:,:,idx, off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = xyz_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end]
            mask_pair_off[:,:,idx, off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = mask_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end]
            if hsv_pair is not None:
                hsv_pair_off[:,:,idx, off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = hsv_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end]

    def calc_inp_from_dense(self, xyz_pair, mask_pair, hsv_pair=None):
        xyz_ell = self.geo_scale
        hsv_ell = 0.4

        half_w = 4
        half_h = 4
        
        height = xyz_pair[0].shape[2]
        width = xyz_pair[1].shape[3]
        # zeros_dummy = torch.zeros((self.opt.batch_size, 1, height, width), device=self.device, dtype=xyz_pair[0].dtype)
        # exp_sum = torch.tensor(0, device=self.device, dtype=xyz_pair[0].dtype)

        num_off = (2*half_w+1) * (2*half_h+1)

        # start_time = time.time()
        xyz_pair_off = torch.zeros((self.opt.batch_size, 3, num_off, height, width), device=self.device, dtype=xyz_pair[0].dtype)
        mask_pair_off = torch.zeros((self.opt.batch_size, 1, num_off, height, width), device=self.device, dtype=torch.bool)
        
        if hsv_pair is not None:
            hsv_pair_off = torch.zeros((self.opt.batch_size, 3, num_off, height, width), device=self.device, dtype=hsv_pair[0].dtype)

        if not self.opt.multithread:

            for i in range(-half_h, half_h+1):
                for j in range(-half_w, half_w+1):
                    off_h0_start = max(0, -i)
                    off_h0_end = min(0, -i)
                    off_w0_start = max(0, -j)
                    off_w0_end = min(0, -j)
                    off_h1_start = -off_h0_end
                    off_h1_end = -off_h0_start
                    off_w1_start = -off_w0_end
                    off_w1_end = -off_w0_start
                    
                    idx = (i + half_h) * (2 * half_w + 1) + j + half_w
                    xyz_pair_off[:,:,idx, off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = xyz_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end]
                    mask_pair_off[:,:,idx, off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = mask_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end]
                    if hsv_pair is not None:
                        hsv_pair_off[:,:,idx, off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = hsv_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end]

                    # diff_xyz = torch.zeros((self.opt.batch_size, 1, height, width), device=self.device, dtype=xyz_pair[0].dtype)
                    # diff_xyz[:,:,off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = \
                    #     torch.pow(torch.norm(xyz_pair[0][:,:,off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] - \
                    #         xyz_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end], dim=1 ), 2)
                    # exp_xyz = torch.exp(-diff_xyz / (2*xyz_ell*xyz_ell))
                    # exp = exp_xyz

                    # if hsv_pair is not None:
                    #     diff_hsv = torch.zeros((self.opt.batch_size, 1, height, width), device=self.device, dtype=hsv_pair[0].dtype)
                    #     diff_hsv[:,:,off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = \
                    #         torch.pow(torch.norm(hsv_pair[0][:,:,off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] - \
                    #             hsv_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end], dim=1 ), 2)
                    #     exp_hsv = torch.exp(-diff_hsv / (2*hsv_ell*hsv_ell))
                    #     exp = torch.mul(exp_xyz, exp_hsv)

                    # mask = mask_pair[0].clone()
                    # mask[:,:,off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] = \
                    #     mask[:,:,off_h0_start : height+off_h0_end, off_w0_start : width+off_w0_end] & \
                    #         mask_pair[1][:,:,off_h1_start : height+off_h1_end, off_w1_start : width+off_w1_end]

                    # exp = torch.where(mask, exp, zeros_dummy)
                    # exp_sum = exp_sum + exp.sum()
        
        else:
            threads = []
            # xyz_pair_off_l = [None] * (2*half_h+1)
            # mask_pair_off_l = [None] * (2*half_h+1)
            # hsv_pair_off_l = [None] * (2*half_h+1)
            xyz_pair_off_l = torch.split(xyz_pair_off, 2*half_w+1, dim=2)
            mask_pair_off_l = torch.split(mask_pair_off, 2*half_w+1, dim=2)
            if hsv_pair is not None:
                hsv_pair_off_l = torch.split(hsv_pair_off, 2*half_w+1, dim=2)

            for zero_idx, i in enumerate(range(-half_h, half_h+1)):
                # xyz_pair_off_l[zero_idx] = torch.zeros((self.opt.batch_size, 3, 2*half_w+1, height, width), device=self.device, dtype=xyz_pair[0].dtype)
                # mask_pair_off_l[zero_idx] = torch.zeros((self.opt.batch_size, 1, 2*half_w+1, height, width), device=self.device, dtype=torch.bool)
                # if hsv_pair is not None:
                #     hsv_pair_off_l[zero_idx] = torch.zeros((self.opt.batch_size, 3, 2*half_w+1, height, width), device=self.device, dtype=hsv_pair[0].dtype)

                x = threading.Thread(target=self.create_moving_slice, args=(i, half_h, half_w, xyz_pair_off_l[zero_idx], mask_pair_off_l[zero_idx], hsv_pair_off_l[zero_idx], xyz_pair, mask_pair, hsv_pair, height, width ) )
                threads.append(x)
                x.start()
            for x in threads:
                x.join()
                
            xyz_pair_off = torch.cat(xyz_pair_off_l, dim=2)
            mask_pair_off = torch.cat(mask_pair_off_l, dim=2)
            hsv_pair_off = torch.cat(hsv_pair_off_l, dim=2)

        # end_time = time.time()
        # print("time elapsed", end_time-start_time)

        diff_xyz = torch.exp( -torch.pow(torch.norm(xyz_pair[0].unsqueeze(2) - xyz_pair_off, dim=1), 2) / (2*xyz_ell*xyz_ell) )
        diff = diff_xyz
        if hsv_pair is not None:
            diff_hsv = torch.exp( -torch.pow(torch.norm(hsv_pair[0].unsqueeze(2) - hsv_pair_off, dim=1), 2) / (2*hsv_ell*hsv_ell) )
            diff = torch.mul(diff_xyz, diff_hsv)

        zeros_dummy = torch.zeros_like(diff)
        diff = torch.where(mask_pair_off, diff, zeros_dummy)

        exp_mask_halfsum = diff.sum(dim=2)
        zeros_dummy = torch.zeros_like(exp_mask_halfsum)
        exp_mask_halfsum = torch.where(mask_pair[0], exp_mask_halfsum, zeros_dummy)

        exp_sum = exp_mask_halfsum.sum()

        return exp_sum 

    def gen_innerp_dense(self, inputs, outputs):
        id_pairs = [(0,0), (1,1), (-1,-1), (0,-1), (0,1)]
        gt_pairs = [(True, True), (True, False), (False, False)]
        innerps = {}
        for id_pair in id_pairs: #self.opt.frame_ids:
            # i, j = id_pair
            for scale in self.opt.scales:
                for gt_pair in gt_pairs:
                    xyz1 = [None] * 2
                    hsv = [None] * 2
                    mask = [None]*2
                    for k in range(2):
                        xyz1[k], hsv[k], mask[k] = self.get_xyz_rgb_pair(id_pair[k], scale, inputs, outputs, gt=gt_pair[k])
                    xyz_aligned = self.get_xyz_aligned(id_pair, xyz1, outputs)
                    
                    innerps[(id_pair, scale, gt_pair)] = self.calc_inp_from_dense(xyz_aligned, mask, hsv)
                    
        return innerps

    def gen_cvo_loss_dense(self, innerps, id_pair, scale, gt_pair):
        i,j = id_pair
        gt_i, gt_j = gt_pair
        inp = innerps[(id_pair, scale, gt_pair)]
        if i == j:
            f_dist = torch.tensor(0, device=self.device, dtype=inp.dtype)
            cos_sim = torch.tensor(0, device=self.device, dtype=inp.dtype)
        else:
            inp_ii = innerps[((i,i), scale, (gt_i, gt_i))]
            inp_jj = innerps[((j,j), scale, (gt_j, gt_j))]
            f_dist = inp_ii + inp_jj - 2 * inp
            cos_sim = 1 - inp/torch.sqrt( inp_ii * inp_jj )
            # cos_sim = 1 - inp/torch.sqrt( torch.max(inp_ii * inp_jj, torch.zeros_like(inp_ii)+1e-7) )
        
        return inp, f_dist, cos_sim

    def generate_depths_pred(self, inputs, outputs, outputs_others=None):
        for scale in self.opt.scales:
            ## depth from disparity
            disp = outputs[("disp", scale)]
            depth, source_scale = self.from_disp_to_depth(disp, scale)
            outputs[("depth", 0, scale)] = depth

            if self.opt.cvo_loss:
                # self.gen_pcl_gt(inputs, outputs, disp, scale, 0)
                depth_curscale, _ = self.from_disp_to_depth(disp, scale, force_multiscale=True)
                outputs[("depth_scale", 0, scale)] = depth_curscale

            for frame_id in self.opt.frame_ids[1:]:
                ## depth from disparity
                if self.opt.cvo_loss:# and frame_id == -1:
                    assert outputs_others is not None, "no disparity prediction of other images!"
                    disp = outputs_others[frame_id][("disp", scale)]
                    
                    # self.gen_pcl_gt(inputs, outputs, disp, scale, frame_id, T_inv) 
                    depth_curscale, _ = self.from_disp_to_depth(disp, scale, force_multiscale=True)
                    outputs[("depth_scale", frame_id, scale)] = depth_curscale         

    # def generate_images_pred(self, inputs, outputs):
    ## ZMH: add depth output of other images
    def generate_images_pred(self, inputs, outputs, outputs_others=None):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            ### ZMH: generate depth of host image from current scale disparity estimation
            disp = outputs[("disp", scale)]
            if disp.requires_grad:
                disp.register_hook(lambda grad: recall_grad("disp", grad))

            depth, source_scale = self.from_disp_to_depth(disp, scale)
            outputs[("depth", 0, scale)] = depth
            ##----------------------
            # if self.opt.v1_multiscale or force_multiscale:
            #     source_scale = scale
            # else:
            #     source_scale = 0
            # depth = outputs[("depth", 0, scale)]


            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])

            if self.opt.cvo_loss:
                # self.gen_pcl_gt(inputs, outputs, disp, scale, 0)
                depth_curscale, _ = self.from_disp_to_depth(disp, scale, force_multiscale=True)
                outputs[("depth_scale", 0, scale)] = depth_curscale
                outputs[("depth_wrap", 0, scale)] = depth_curscale
                cam_points_curscale = self.backproject_depth[scale](
                    depth_curscale, inputs[("inv_K", scale)])
                
                if self.opt.cvo_loss_dense:
                    if not self.opt.dense_flat_grid:
                        self.get_xyz_dense(0, scale, inputs, outputs)
                    else:
                        self.get_grid_flat(0, scale, inputs, outputs)
                else:
                    masks = self.gen_pcl_wrap_host(inputs, outputs, scale)

            ### ZMH: the neighboring images (either the stereo counterpart or the previous/next image)
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if T.requires_grad:
                    T.register_hook(lambda grad: recall_grad("T", grad))

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                T_inv = torch.inverse(T)
                
                if self.show_range and scale == self.opt.scales[0]:
                    print("frame", frame_id, "\n", T)

                ## ZMH: what here it is doing: 
                ## using depth of host frame 0, reprojecting to another frame i, reconstruct host frame 0 using reprojection coords and frame i's pixels.
                ## The difference between true host frame 0 and reconstructed host frame 0 is the photometric error

                ### ZMH: px = T_x0 *p0
                ### ZMH: therefore the T is the pose of host relative to frame x
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if self.torch_version <= 1.2:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border")
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border", align_corners=True)
                ## ZMH: generate depth of other images
                if self.opt.cvo_loss:# and frame_id == -1:

                    assert outputs_others is not None, "no disparity prediction of other images!"
                    disp = outputs_others[frame_id][("disp", scale)]
                    
                    # self.gen_pcl_gt(inputs, outputs, disp, scale, frame_id, T_inv) 
                    depth_curscale, _ = self.from_disp_to_depth(disp, scale, force_multiscale=True)
                    outputs[("depth_scale", frame_id, scale)] = depth_curscale         

                    if self.opt.cvo_loss_dense:
                        if not self.opt.dense_flat_grid:
                            self.get_xyz_dense(frame_id, scale, inputs, outputs)
                        else:
                            self.get_grid_flat(frame_id, scale, inputs, outputs)
                    else:
                        ## ZMH: sample in adjacent frames
                        pix_coords_curscale = self.project_3d[scale](
                            cam_points_curscale, inputs[("K", scale)], T)
                        outputs[("sample_wrap", frame_id, scale)] = pix_coords_curscale

                        if self.torch_version <= 1.2:
                            outputs[("depth_wrap", frame_id, scale)] = F.grid_sample(
                                outputs[("depth_scale", frame_id, scale)],
                                outputs[("sample_wrap", frame_id, scale)],
                                padding_mode="border")

                            outputs[("color_wrap", frame_id, scale)] = F.grid_sample(
                                inputs[("color", frame_id, scale)],
                                outputs[("sample_wrap", frame_id, scale)],
                                padding_mode="border")

                            outputs[("uv_wrap", frame_id, scale)] = F.grid_sample(
                                self.backproject_depth[scale].id_coords.unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1),
                                outputs[("sample_wrap", frame_id, scale)],
                                padding_mode="border")          # the first argument: B*2*H*W, 2nd arg: B*H*W*2, output: B*2*H*W
                        else:
                            outputs[("depth_wrap", frame_id, scale)] = F.grid_sample(
                                outputs[("depth_scale", frame_id, scale)],
                                outputs[("sample_wrap", frame_id, scale)],
                                padding_mode="border", align_corners=True)

                            outputs[("color_wrap", frame_id, scale)] = F.grid_sample(
                                inputs[("color", frame_id, scale)],
                                outputs[("sample_wrap", frame_id, scale)],
                                padding_mode="border", align_corners=True)

                            outputs[("uv_wrap", frame_id, scale)] = F.grid_sample(
                                self.backproject_depth[scale].id_coords.unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1),
                                outputs[("sample_wrap", frame_id, scale)],
                                padding_mode="border", align_corners=True)          # the first argument: B*2*H*W, 2nd arg: B*H*W*2, output: B*2*H*W
                        
                        self.gen_pcl_wrap_other(inputs, outputs, scale, frame_id, T_inv, masks)

                    # ### ZMH: transform the points in host frame to source frame
                    # cam_pts_trans = torch.matmul(T, cam_points)
                    # ### ZMH: Flatten color matrix
                    # color_ori = inputs["color_aug", 0, 0].view(cam_points.shape[0], 3, -1)
                    # color_other = inputs["color_aug", frame_id, 0].view(cam_points.shape[0], 3, -1)
                    # print(cam_points.shape) # B*4*N
                    # print(cam_points.dtype)
                    # draw3DPts(cam_pts_trans.detach()[:,:3,:], pcl_2=cam_points_other.detach()[:,:3,:], color_1=color_ori.detach(), color_2=color_other.detach())

                    # ### ZMH: visualize the grount truth point cloud
                    # for ib in range(self.opt.batch_size):
                    #     draw3DPts(outputs[("xyz_gt", 0, 0)][ib].detach()[:,:3,:], pcl_2=outputs[("xyz_gt", frame_id, scale)][ib].detach()[:,:3,:], 
                    #         color_1=outputs[("rgb_gt", 0, 0)][ib].detach(), color_2=outputs[("rgb_gt", frame_id, scale)][ib].detach())

                    if not self.opt.cvo_loss_dense:
                        for ib in range(self.opt.batch_size):
                            print(outputs[("rgb_gt", 0, 0)][ib]*255)
                            visualize_pcl(outputs[("xyz_gt", 0, 0)][ib].detach()[:,:3,:], rgb=outputs[("rgb_gt", 0, 0)][ib].detach() )
                    

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    # def func_inner_prod(self, gramian_p, gramian_c):
    #     """
    #     Calculate the inner product of two functions from gramian matrix of two point clouds
    #     """
    #     prod = torch.sum(gramian_p * gramian_c)
    #     return prod

    def loss_from_inner_prod(self, inner_prods):
        f1_f2_dist = inner_prods[(0,0)] + inner_prods[(1,1)] - 2 * inner_prods[(0,1)]
        cos_similarity = 1 - inner_prods[(0,1)] / torch.sqrt( inner_prods[(0,0)] * inner_prods[(1,1)] )
        # cos_similarity = 1 - inner_prods[(0,1)] / torch.sqrt( torch.max(inner_prods[(0,0)] * inner_prods[(1,1)], torch.zeros_like(inner_prods[(0,0)])+1e-7 ) )

        return f1_f2_dist, cos_similarity

    def inner_prod_from_gramian(self, gramians ):
        """
        Calculate function distance loss and cosine similarity by multiplying the gramians in all domains together for a specific pair
        """
        inner_prods = {}
        list_of_ij = [(0,0), (1,1), (0,1)]

        for ij in list_of_ij:
            gramian_list = [gramian[ij] for gramian in gramians.values() ]
            # gramian_stack = torch.stack(gramian_list, dim=3)
            # inner_prods[ij] = torch.sum(torch.prod(gramian_stack, dim=3))
            if len(gramian_list) == 1:
                inner_prods[ij] = torch.sum(gramian_list[0])
            else:
                inner_p = gramian_list[0] * gramian_list[1]
                for k in range(2,len(gramian_list)):
                    inner_p = inner_p * gramian_list[k]
                inner_prods[ij] = torch.sum(inner_p)
            if self.opt.normalize_inprod_over_pts:
                inner_prods[ij] = inner_prods[ij] / (gramian_list[0].shape[1] * gramian_list[0].shape[2])

        return inner_prods

    def cvo_gramian(self, vectors, dist_coef, normalize_mode="ori"):
        """
        Compute the gramian matix for a pair of vectors in a specific domain (xyz, hsv, etc.)
        Output: {(0,0): _, (1,1): _, (0,1): _}
        """
        # B*C*N
        vec_local = {}
        vec_local[0] = vectors[0]
        vec_local[1] = vectors[1]
        
        # print(vec_local[0].std(dim=2, keepdim=True) )
        # print(vec_local[1].std(dim=2, keepdim=True) )
        
        if normalize_mode == "sep":
            vec_local[0] = vec_local[0] - vec_local[0].mean(dim=2, keepdim=True).expand_as(vec_local[0])
            vec_local[0] = vec_local[0] / vec_local[0].std(dim=2, keepdim=True).expand_as(vec_local[0])
            vec_local[1] = vec_local[1] - vec_local[1].mean(dim=2, keepdim=True).expand_as(vec_local[1])
            vec_local[1] = vec_local[1] / vec_local[1].std(dim=2, keepdim=True).expand_as(vec_local[1])
        elif normalize_mode == "tog":
            v12 = torch.cat((vec_local[0], vec_local[1]), dim=2)
            v12 = v12 - v12.mean(dim=2, keepdim=True).expand_as(v12)
            v12 = v12 / v12.std(dim=2, keepdim=True).expand_as(v12)
            vec_local[0] = v12[:,:,:vec_local[0].shape[2]]
            vec_local[1] = v12[:,:,vec_local[0].shape[2]:]

        gramians = {}
        list_of_ij = [(0,0), (1,1), (0,1)]
        for ij in list_of_ij:
            (i,j) = ij
            gramians[ij] = kern_mat(vec_local[i], vec_local[j], dist_coef)
        
        return gramians

    def compute_cvo_loss_with_options(self, vector_to_cvo, items_to_cal_gram, dist_coefs, norm_tags):

        # ### A way to save memory: 
        # ### Calculate all gramians altogether. 
        # thre_t = 8.315e-3
        # # thre_d = -2.0 * dist_coef * dist_coef * np.log(thre_t)
        # inner_prods = {}
        # ij_list = [(0,0), (1,1), (0,1)]
        # for ij in ij_list:
        #     i,j  = ij
        #     inner_prods[ij] = torch.zeros([], device=self.device, dtype=torch.float32)
        #     # gramian_all = torch.zeros((1, vector_to_cvo[items_to_cal_gram[0]][0].shape[2]), device=self.device, dtype=torch.float32)
        #     # zero_dummy = torch.zeros((1, vector_to_cvo[items_to_cal_gram[0]][i].shape[2]), device=self.device, dtype=torch.float32)
        #     for k in range(vector_to_cvo[items_to_cal_gram[0]][j].shape[2]):
        #         gramian_iter = torch.ones((1, vector_to_cvo[items_to_cal_gram[0]][i].shape[2]), device=self.device, dtype=torch.float32)
        #         for item in items_to_cal_gram:
        #             vec0 = vector_to_cvo[item][i]
        #             vec1 = vector_to_cvo[item][j]
        #             diff = torch.norm(vec0 - vec1[..., k:k+1].expand_as(vec0), dim=1)
        #             diff_exp = torch.exp(-torch.pow(diff, 2) / (2*dist_coefs[item]*dist_coefs[item]) )
        #             # diff_exp = torch.where(diff_exp >= thre_t, diff_exp, torch.zeros_like(diff_exp) )
        #             # diff_exp = torch.where(diff_exp >= thre_t, diff_exp, zero_dummy )
        #             gramian_iter = torch.mul(gramian_iter, diff_exp)
        #         inner_prods[ij] = inner_prods[ij] + gramian_iter.sum()

        ### Calculate gramians
        gramians = {}
        for item in items_to_cal_gram: # for item in vector_to_cvo:
            gramians[item] = self.cvo_gramian(vector_to_cvo[item], dist_coefs[item], normalize_mode=norm_tags[item])

        ### Calculate inner product
        inner_prods = self.inner_prod_from_gramian(gramians )


        ### Calculate CVO loss
        f1_f2_dist, cos_similarity = self.loss_from_inner_prod( inner_prods )
        
        return f1_f2_dist, cos_similarity, -inner_prods[(0,1)]

    def name_loss_from_norm_options(self, items_to_cal_gram, norm_tags):
        item_tags = {}
        name_loss = ""
        for item in items_to_cal_gram:
            name_loss = name_loss + item + "_" + norm_tags[item] + "_"
        return name_loss
        
    def compute_cvo_loss(self, vector_to_cvo):

        if "rgb" in vector_to_cvo:
            vector_to_cvo["hsv"] = {}
            for i in range(2):
                vector_to_cvo["hsv"][i] = rgb_to_hsv(vector_to_cvo["rgb"][i], flat=True )
            
        items_to_calculate_gram = ["hsv", "xyz"]
        # items_to_calculate_gram = ["xyz"]

        dist_coefs = {}
        # for item in items_to_calculate_gram:
        #     dist_coefs[item] = 0.1
        dist_coefs["xyz"] = self.geo_scale
        if "rgb" in vector_to_cvo:
            dist_coefs["hsv"] = 0.4

        f1_f2_dist = {}
        cos_similarity = {}
        inner_prod = {}

        norm_tags = {}
        # norm_tags["hsv"] = "tog" # or "tog" "ori"
        # norm_tags["xyz"] = "tog" # or "ori"

        # name_loss = self.name_loss_from_norm_options(items_to_calculate_gram, norm_tags)
        # f1_f2_dist[name_loss], cos_similarity[name_loss] = self.compute_cvo_loss_with_options(vector_to_cvo, items_to_calculate_gram, dist_coefs, norm_tags)

        # norm_tags["hsv"] = "tog" # or "tog" "ori"
        # norm_tags["xyz"] = "ori" # or "ori"

        # name_loss = self.name_loss_from_norm_options(items_to_calculate_gram, norm_tags)
        # f1_f2_dist[name_loss], cos_similarity[name_loss] = self.compute_cvo_loss_with_options(vector_to_cvo, items_to_calculate_gram, dist_coefs, norm_tags)

        if "rgb" in vector_to_cvo:
            norm_tags["hsv"] = "ori" # or "tog" "ori"
        norm_tags["xyz"] = "ori" # or "ori"

        name_loss = self.name_loss_from_norm_options(items_to_calculate_gram, norm_tags)
        f1_f2_dist[name_loss], cos_similarity[name_loss], inner_prod[name_loss] = self.compute_cvo_loss_with_options(vector_to_cvo, items_to_calculate_gram, dist_coefs, norm_tags)

        return f1_f2_dist, cos_similarity, inner_prod

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_disp_losses(self, inputs, outputs, outputs_others):
        disp_losses = {}
        for scale in self.opt.scales:
            for frame_id in self.opt.frame_ids:
                if frame_id == 0:
                    disp = outputs[("disp", scale)]
                else:
                    disp = outputs_others[frame_id][("disp", scale)]
                depth_gt = inputs[("depth_gt_scale", frame_id, scale)]
                disp_gt = depth_to_disp(depth_gt, self.opt.min_depth, self.opt.max_depth)
                # print("disp_gt range", torch.min(disp_gt), torch.max(disp_gt))
                if frame_id == self.opt.frame_ids[0]:
                    disp_losses[scale] = torch.tensor(0, dtype=torch.float32, device=self.device)
                disp_losses[scale] += self.compute_disp_loss(disp, disp_gt)
        return disp_losses

    def compute_disp_loss(self, disp, disp_gt):
        mask = disp_gt > 0
        disp_gt_masked = disp_gt[mask] # becomes a 1-D tensor
        disp_masked = disp[mask]

        disp_error = torch.mean(torch.abs(disp_masked - disp_gt_masked))
        return disp_error


    def compute_losses(self, inputs, outputs, outputs_others):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        ### ZMH: CVO loss
        # total_cvo_loss = 0
        # total_cos_loss = 0

        if self.opt.cvo_loss_dense:
            if not self.opt.dense_flat_grid:
                innerps = self.gen_innerp_dense(inputs, outputs)
            else:
                n_pts_dict = self.concat_flat(outputs) ## ZMH: current version fixed the memory leak
                if self.opt.use_normal_v2:        ## ZMH: comment this line for memory leak debug
                    self.get_grid_flat_normal(outputs, n_pts_dict)
                

                if self.step % 4000 == 0: ## ZMH: comment this line for memory leak debug! 
                    self.save_pcd(outputs, n_pts_dict)

                innerps, dists, coss = self.get_innerp_from_grid_flat(outputs, n_pts_dict)
                # innerps, dists, coss = self.get_innerp_from_grid_flat_dummy(outputs) ## ZMH: for memory leak debug 
                self.reg_cvo_to_loss(losses, innerps, dists, coss)
                # self.reg_cvo_to_loss_dummy(losses, innerps, dists, coss)  ## ZMH: for memory leak debug 


        if self.opt.sup_cvo_pose_lidar: # not self.train_flag or (this is used for evaluating cvo pose loss in baseline mode)
            if self.opt.cvo_loss_dense:
                if not self.opt.dense_flat_grid:
                    losses["loss_pose/cos_sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
                    losses["loss_pose/cvo_sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
                    for frame_id in self.opt.frame_ids[1:]:
                        for scale in self.opt.scales:
                            inp, f_dist, cos_sim = self.gen_cvo_loss_dense(innerps, (0, frame_id), scale, (True, True) )
                            losses["loss_pose/cvo_s{}_f{}".format(scale, frame_id)] = f_dist
                            losses["loss_pose/cos_s{}_f{}".format(scale, frame_id)] = cos_sim
                            losses["loss_pose/inp_s{}_f{}".format(scale, frame_id)] = inp

                            losses["loss_pose/cos_sum"] += cos_sim
                            losses["loss_pose/cvo_sum"] += f_dist
            else:
                self.calc_cvo_pose_loss(inputs, outputs, losses)

            

        disp_losses = self.compute_disp_losses(inputs, outputs, outputs_others)
        for scale in self.opt.scales:
            losses["loss_disp/{}".format(scale)] = disp_losses[scale]
        
        if not self.train_flag or self.opt.supervised_by_gt_depth:
            if self.opt.cvo_loss and not (self.opt.cvo_loss_dense and self.opt.dense_flat_grid):
                losses["loss_cvo/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
                losses["loss_cos/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
                losses["loss_inp/sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            ### ZMH: CVO loss
            
            if not self.train_flag or self.opt.supervised_by_gt_depth:
                if self.opt.cvo_loss:
                    # cvo_losses = []
                    # cos_losses = []
                    # cvo_loss = torch.tensor(0., dtype = reprojection_losses.dtype, device=reprojection_losses.device)
                    # cos_loss = torch.tensor(0., dtype = reprojection_losses.dtype, device=reprojection_losses.device)

                    # if True: #outputs[("xyz_in_host", 0, scale)][:,0:3,:].shape[2] <= 10000:
                    # gt_mode = self.opt.supervised_by_gt_depth:
                    # for gt_mode in [True, False]:
                    # if outputs[("xyz_in_host", 0, scale)][:,0:3,:].shape[2] <= 5000:
                    ### calculate cvo_loss only when the image scale is not too large
                    
                    # for frame_id in self.opt.frame_ids[1:]:
                    #     if gt_mode:
                    #         xyz_0 = outputs[("xyz_gt", frame_id, 0)]
                    #         rgb_0 = outputs[("rgb_gt", frame_id, 0)]
                    #     else:
                    #         xyz_0 = outputs[("xyz_in_host", 0, scale)]
                    #         rgb_0 = outputs[("rgb_in_host", 0, scale)]

                    #     xyz_1 = outputs[("xyz_in_host", frame_id, scale)]
                    #     rgb_1 = outputs[("rgb_in_host", frame_id, scale)]
                    
                    if self.opt.cvo_loss_dense:
                        if not self.opt.dense_flat_grid:
                            for frame_id in self.opt.frame_ids[1:]:
                                inp, f_dist, cos_sim = self.gen_cvo_loss_dense(innerps, (0, frame_id), scale, (True, False) )
                                
                                losses["loss_cvo/{}_s{}_f{}".format(True, scale, frame_id)] = f_dist
                                losses["loss_cos/{}_s{}_f{}".format(True, scale, frame_id)] = cos_sim
                                losses["loss_inp/{}_s{}_f{}".format(True, scale, frame_id)] = inp
                                
                                losses["loss_cvo/sum"] += f_dist
                                losses["loss_cos/sum"] += cos_sim
                                losses["loss_inp/sum"] += inp

                    else:
                        for gt_mode in [True]:# [True, False]:
                            for frame_id in self.opt.frame_ids:
                                cvo_losses = {}
                                cos_losses = {}
                                innerp_losses = {}
                                # if frame_id == 1:
                                #     continue
                                if not gt_mode:
                                    if frame_id==self.opt.frame_ids[0]:
                                        xyz_0 = outputs[("xyz_gt", 0, scale)]
                                        rgb_0 = outputs[("rgb_gt", 0, scale)]
                                    else:
                                        xyz_0 = outputs[("xyz_in_host", frame_id, scale)]
                                        rgb_0 = outputs[("rgb_in_host", frame_id, scale)]

                                    xyz_1 = outputs[("xyz_in_host", 0, scale)]
                                    rgb_1 = outputs[("rgb_in_host", 0, scale)]
                                else:
                                    xyz_0 = outputs[("xyz_in_host", frame_id, scale)]
                                    rgb_0 = outputs[("rgb_in_host", frame_id, scale)]
                                    xyz_1 = outputs[("xyz_gt", 0, scale)]
                                    rgb_1 = outputs[("rgb_gt", 0, scale)]

                                for ib in range(self.opt.batch_size):
                                    vector_to_cvo = {}
                                    vector_to_cvo["xyz"] = {}
                                    vector_to_cvo["rgb"] = {}
                                    samp_pt = 3500 #4000  # the original number of points are about 5k, 1k, 0.3k, 0.1k (gen_pcl_gt masked out points without gt measurements)
                                    # print("pcl 0", "ib", ib, "frame_id", frame_id, "scale", scale, "size", xyz_0[ib].shape[-1])
                                    # print("pcl 1", "ib", ib, "frame_id", frame_id, "scale", scale, "size", xyz_1[ib].shape[-1])
                                    if xyz_0[ib].shape[-1] > samp_pt:
                                        # print('gt sampling!', scale)
                                        num_from_gt = xyz_0[ib].shape[-1]
                                        idx_gt = torch.randperm(num_from_gt)[:samp_pt]
                                        vector_to_cvo["xyz"][0] = xyz_0[ib][:,0:3,idx_gt] # self.opt.scales[-1]
                                        vector_to_cvo["rgb"][0] = rgb_0[ib][...,idx_gt]
                                    else:
                                        vector_to_cvo["xyz"][0] = xyz_0[ib][:,0:3,:]
                                        vector_to_cvo["rgb"][0] = rgb_0[ib]
                                    
                                    if xyz_1[ib].shape[-1] > samp_pt:
                                        # print('est sampling!', scale)
                                        num_from_est = xyz_1[ib].shape[-1]
                                        idx_est = torch.randperm(num_from_est)[:samp_pt]
                                        vector_to_cvo["xyz"][1] = xyz_1[ib][:,0:3,idx_est]
                                        vector_to_cvo["rgb"][1] = rgb_1[ib][...,idx_est]
                                        # vector_to_cvo["xyz"][1] = xyz_1[ib][:,0:3,idx_gt]
                                        # vector_to_cvo["rgb"][1] = rgb_1[ib][...,idx_gt]
                                    else:
                                        vector_to_cvo["xyz"][1] = xyz_1[ib][:,0:3,:]
                                        vector_to_cvo["rgb"][1] = rgb_1[ib]

                                    # if self.show_range and scale == 0 and frame_id == 1:
                                    #     print("xyz gt 0 min",  torch.min(vector_to_cvo["xyz"][0], dim=2)[0]) # x: [-70,70], y: [-3, 15], z: [-0.3, 80]
                                    #     print("xyz gt 0 max",  torch.max(vector_to_cvo["xyz"][0], dim=2)[0]) # x: [-26, 13], y: [-3, 2], z: [5, 76]
                                    #     print("xyz gt 1 min",  torch.min(vector_to_cvo["xyz"][1], dim=2)[0]) # 
                                    #     print("xyz gt 1 max",  torch.max(vector_to_cvo["xyz"][1], dim=2)[0])
                                    
                                    cvo_loss, cos_loss, innerp_loss = self.compute_cvo_loss( vector_to_cvo )

                                    for item in cvo_loss:
                                        if ib == 0:
                                            cvo_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                                        cvo_losses[item] += cvo_loss[item] / ( self.opt.batch_size )
                                    for item in cos_loss:
                                        if ib == 0 :
                                            cos_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                                        cos_losses[item] += cos_loss[item] / (self.opt.batch_size )
                                    for item in innerp_loss:
                                        if ib == 0:
                                            innerp_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                                        innerp_losses[item] += innerp_loss[item] / ( self.opt.batch_size )

                                    # for item in cvo_loss:
                                    #     if ib == 0 and frame_id == self.opt.frame_ids[0]:
                                    #         cvo_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                                    #     cvo_losses[item] += cvo_loss[item] / ( (len(self.opt.frame_ids)-1) * self.opt.batch_size )
                                    # for item in cos_loss:
                                    #     if ib == 0 and frame_id == self.opt.frame_ids[0]:
                                    #         cos_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                                    #     cos_losses[item] += cos_loss[item] / ((len(self.opt.frame_ids)-1) * self.opt.batch_size )
                                    # for item in innerp_loss:
                                    #     if ib == 0 and frame_id == self.opt.frame_ids[0]:
                                    #         innerp_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                                    #     innerp_losses[item] += innerp_loss[item] / ((len(self.opt.frame_ids)-1) * self.opt.batch_size )
                            
                                for item in cvo_loss:
                                    losses["loss_cvo/{}_{}_s{}_f{}".format(item, gt_mode, scale, frame_id)] = cvo_losses[item]
                                    losses["loss_cos/{}_{}_s{}_f{}".format(item, gt_mode, scale, frame_id)] = cos_losses[item]
                                    losses["loss_inp/{}_{}_s{}_f{}".format(item, gt_mode, scale, frame_id)] = innerp_losses[item]

                                    if gt_mode :#and scale >=2:
                                        losses["loss_cvo/sum"] += cvo_losses[item]
                                        losses["loss_cos/sum"] += cos_losses[item]
                                        losses["loss_inp/sum"] += innerp_losses[item]



            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda(self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            
            # if self.opt.cvo_loss:
            #     total_cvo_loss += cvo_loss
            #     losses["loss_cvo/{}".format(scale)] = cvo_loss
            #     total_cos_loss += cos_loss
            #     losses["loss_cos/{}".format(scale)] = cos_loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        # if self.opt.cvo_loss:
        #     for item in cvo_loss:
        #         losses["loss_cvo/{}".format(item)] = cvo_losses[item]
        #         losses["loss_cos/{}".format(item)] = cos_losses[item]

        return losses

    def calc_cvo_pose_loss(self, inputs, outputs, losses):
        """
        This is to supervise the pose prediction by calculating the CVO loss between two lidar point clouds.
        """

        losses["loss_pose/cos_sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        losses["loss_pose/cvo_sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        # losses["loss_pose/inp_sum"] =  torch.tensor(0, dtype=torch.float32, device=self.device)
        for frame_id in self.opt.frame_ids[1:]:
            for scale in self.opt.scales:

                cvo_losses = {}
                cos_losses = {}
                innerp_losses = {}

                for ib in range(self.opt.batch_size):
                    if frame_id == "s":
                        T = inputs["stereo_T"]
                    else:
                        T = outputs[("cam_T_cam", 0, frame_id)][ib:ib+1] #T_i0
                        
                    vector_to_cvo = {}
                    vector_to_cvo["xyz"] = {}
                    # vector_to_cvo["xyz"][0] = torch.matmul(T, inputs[("velo_gt", 0)][ib].transpose(1,2))[:,:3,:]
                    # vector_to_cvo["xyz"][1] = inputs[("velo_gt", frame_id)][ib].transpose(1,2)[:,:3,:]
                    vector_to_cvo["xyz"][0] = torch.matmul(T, outputs[("xyz_gt", 0, scale)][ib])[:,:3,:]
                    vector_to_cvo["xyz"][1] = outputs[("xyz_gt", frame_id, scale)][ib][:,:3,:]

                    vector_to_cvo["rgb"] = {}
                    vector_to_cvo["rgb"][0] = outputs[("rgb_gt", 0, scale)][ib]
                    vector_to_cvo["rgb"][1] = outputs[("rgb_gt", frame_id, scale)][ib]

                    # print("xyz gt 0 min",  torch.min(vector_to_cvo["xyz"][0], dim=2)) # x: [-70,70], y: [-3, 15], z: [-0.3, 80]
                    # print("xyz gt 0 max",  torch.max(vector_to_cvo["xyz"][0], dim=2))
                    # print("xyz gt 1 min",  torch.min(vector_to_cvo["xyz"][1], dim=2))
                    # print("xyz gt 1 max",  torch.max(vector_to_cvo["xyz"][1], dim=2))
                    
                    # print("# of points: 0", vector_to_cvo["xyz"][0].shape)
                    # print("# of points: 1", vector_to_cvo["xyz"][1].shape)
                    ## ZMH: typically before sampling there are about 60k~65k points, a 640*192 image is about twice of that. 
                    
                    samp_num = 5000
                    for k in range(2):
                        num_el = vector_to_cvo["xyz"][k].shape[2]
                        if num_el > samp_num:
                            perm = torch.randperm( num_el )
                            idx = perm[:samp_num]
                            vector_to_cvo["xyz"][k] = vector_to_cvo["xyz"][k][:,:,idx]
                            vector_to_cvo["rgb"][k] = vector_to_cvo["rgb"][k][:,:,idx]
                    
                    # print("xyz pose 0 min",  torch.min(vector_to_cvo["xyz"][0], dim=2)[0]) # x: [-70,70], y: [-3, 15], z: [-0.3, 80]
                    # print("xyz pose 0 max",  torch.max(vector_to_cvo["xyz"][0], dim=2)[0])
                    # print("xyz pose 1 min",  torch.min(vector_to_cvo["xyz"][1], dim=2)[0])
                    # print("xyz pose 1 max",  torch.max(vector_to_cvo["xyz"][1], dim=2)[0])
                    # draw3DPts( vector_to_cvo["xyz"][0].detach(),  vector_to_cvo["xyz"][1].detach() )
                    # print("from pose")
                    cvo_loss, cos_loss, innerp_loss = self.compute_cvo_loss( vector_to_cvo )

                    for item in cvo_loss:
                        if ib == 0:
                            cvo_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                        cvo_losses[item] += cvo_loss[item] / ((len(self.opt.frame_ids)-1) * self.opt.batch_size )
                    for item in cos_loss:
                        if ib == 0:
                            cos_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                        cos_losses[item] += cos_loss[item] / ((len(self.opt.frame_ids)-1) * self.opt.batch_size )
                    for item in innerp_loss:
                        if ib == 0:
                            innerp_losses[item] = torch.tensor(0, dtype=torch.float32, device=self.device)
                        innerp_losses[item] += innerp_loss[item] / ((len(self.opt.frame_ids)-1) * self.opt.batch_size )

                    
                    for item in cvo_loss:
                        losses["loss_pose/cvo_{}_s{}_f{}".format(item, scale, frame_id)] = cvo_losses[item]
                        losses["loss_pose/cos_{}_s{}_f{}".format(item, scale, frame_id)] = cos_losses[item]
                        losses["loss_pose/inp_{}_s{}_f{}".format(item, scale, frame_id)] = innerp_losses[item]

                        losses["loss_pose/cvo_sum"] += cvo_losses[item]
                        losses["loss_pose/cos_sum"] += cos_losses[item]
                        # losses["loss_pose/inp_sum"] += innerp_losses[item]
                    

        # return cvo_losses, cos_losses, innerp_losses
            

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]

        if "TUM" not in self.opt.dataset:
            depth_pred = torch.clamp(F.interpolate(
                depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        else:
            depth_pred = torch.clamp(F.interpolate(
                depth_pred, [480, 640], mode="bilinear", align_corners=False), 1e-3, 80)

        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        if "TUM" not in self.opt.dataset:
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        if self.writers is None:
            return
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        if mode == "val_set":
            return

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    if frame_id == 0: ## added by ZMH
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                # if s == 0:
                    # print('shape 1', outputs[("disp", s)][j].shape)
                disp = depth_to_disp(inputs[("depth_gt_scale", 0, s)][j], self.opt.min_depth, self.opt.max_depth)
                # disp = disp.squeeze(1)
                # print('shape 2', disp.shape)
                writer.add_image(
                    "disp_{}/gt_{}".format(s, j),
                    normalize_image(disp), self.step)

                writer.add_image(
                    "disp_{}/mask_{}".format(s, j),
                    inputs[("depth_mask", 0, s)][j], self.step)

                if self.opt.mask_samp_as_lidar:
                    writer.add_image(
                        "disp_{}/masksp_{}".format(s, j),
                        inputs[("depth_mask_sp", 0, s)][j], self.step)
                    writer.add_image(
                        "disp_{}/maskgtsp_{}".format(s, j),
                        inputs[("depth_mask_gt_sp", 0, s)][j], self.step)

                if self.opt.use_normal or self.opt.use_normal_v2:
                    writer.add_image(
                        "normal_{}/dir_{}".format(s, j),
                        outputs[("grid_normal_vis", 0, s, 0, False)][j], self.step)
                    writer.add_image(
                        "normal_{}/res_{}".format(s, j),
                        outputs[("grid_nres", 0, s, 0, False)][j], self.step)
                        # normalize_image(outputs[("grid_nres", 0, s, 0, False)][j]), self.step)

                    writer.add_image(
                        "normal_{}/dir_gt_{}".format(s, j),
                        outputs[("grid_normal_vis", 0, s, 0, True)][j], self.step)
                    writer.add_image(
                        "normal_{}/res_gt_{}".format(s, j),
                        outputs[("grid_nres", 0, s, 0, True)][j], self.step)
                        # normalize_image(outputs[("grid_nres", 0, s, 0, True)][j]), self.step)
                

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    if s == 0: ## added by ZMH
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        if self.path_opt is None: ## don't save the file if path_opt is set to None
            return
        if not os.path.exists(self.path_opt):
            os.makedirs(self.path_opt)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(self.path_opt, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        if self.path_model is None:
            return
        save_folder = os.path.join(self.path_model, "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
