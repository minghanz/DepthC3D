# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "tmp"))
                              #    default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "TUM_split", "eigen_zhou_bench_as_val"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "TUM"])
        self.parser.add_argument("--dataset_val",
                                 type=str,
                                 help="dataset to evaluate on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "TUM"]) # ZMH: this option added by me
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--ref_depth",
                                 type=float,
                                 help="ref depth r in r/(r+d)",
                                 default=10.0)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--iters_per_update",
                                 type=int,
                                 help="this value times batch_size is the effective batch size",
                                 default=1) ### ZMH: added
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])
        ## switch for enabling CVO loss
        self.parser.add_argument("--cvo_loss",
                                 help="if set, calculate cvo loss",
                                 action="store_true")
        self.parser.add_argument("--normalize_inprod_over_pts",
                                 help="if set, inner product is divided by the product of total number of points in two pcls",
                                 action="store_true")
        self.parser.add_argument("--supervised_by_gt_depth",
                                 help="if set, the CVO loss is computed by comparing with true depth",
                                 action="store_true")
        self.parser.add_argument("--cvo_as_loss",
                                 help="if set, cvo loss is used for training",
                                 action="store_true")
        self.parser.add_argument("--sup_cvo_pose_lidar",
                                 help="if set, cvo loss is used for supervising pose using lidar points",
                                 action="store_true")
        self.parser.add_argument("--disp_in_loss",
                                 help="if set, disparity L1 difference is used in depth training",
                                 action="store_true")
        self.parser.add_argument("--cvo_loss_dense",
                                 help="if set, calculate cvo loss using dense image",
                                 action="store_true")
        self.parser.add_argument("--multithread",
                                 help="if set, dense cvo loss is calculated multithreaded",
                                 action="store_true")
        self.parser.add_argument("--dense_flat_grid",
                                 help="if set, calculate dense cvo loss using flat-grid correspondence",
                                 action="store_true")
        self.parser.add_argument("--mask_samp_as_lidar",
                                 help="if set, the mask for image is sampled to have the same number of valid points as lidar projection mask",
                                 action="store_true")
        self.parser.add_argument("--use_panoptic",
                                 help="if set, call panoptic network to produce features",
                                 action="store_true")
        self.parser.add_argument("--use_normal",
                                 help="if set, calc normal vectors from depth and used in cvo calculation",
                                 action="store_true")
        self.parser.add_argument("--use_normal_v2",
                                 help="if set, calc normal vectors from depth and used in cvo calculation, using the custom operation PtSampleInGridCalcNormal",
                                 action="store_true")
        self.parser.add_argument("--random_ell",
                                 help="if set, the length scale of geometric kernel is selected randomly following a certain distribution",
                                 action="store_true")
        self.parser.add_argument("--ell_basedist",
                                 type=float, 
                                 help="if not zero, the length scale is proportional to the depth of gt points when the depth is larger than this value. If zero, ell is constant",
                                 default=0)
        self.parser.add_argument("--norm_in_dist",
                                 help="if set, the normal information will be used in exp kernel besides as a coefficient term. Neet use_normal_v2 to be true to be effective",
                                 action="store_true")
        self.parser.add_argument("--res_mag_min",
                                 type=float, 
                                 help="the minimum value for the normal kernel (or viewing it as a coefficient of geometric kernel)",
                                 default=0.1)
        self.parser.add_argument("--res_mag_max",
                                 type=float, 
                                 help="the maximum value for the normal kernel (or viewing it as a coefficient of geometric kernel)",
                                 default=2)
        self.parser.add_argument("--disable_log",
                                 help="if set, no logging of this training trial will be logged to hard drive",
                                 action="store_true")
        self.parser.add_argument("--align_preds",
                                 help="if set, inner prod between predictions of adjacent frames are included in loss",
                                 action="store_true")
        ######## for val_set
        self.parser.add_argument("--val_set_only", 
                                 help="if set, only run val_set on pretrained weights", 
                                 action="store_true")
        self.parser.add_argument("--load_weights_folder_parent", 
                                 type=str,
                                 help="parent path of models to load, needed for val_set_only")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)
        self.parser.add_argument("--cuda_n",
                                 type=int, 
                                 help="which GPU to use",
                                 default=0)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options, rest = self.parser.parse_known_args()
        return self.options, rest
