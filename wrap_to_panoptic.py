import torch
import numpy as np

def to_panoptic(img, ups_cfg): #config.network.pixel_means, [target_size], config.test.max_size
    # 1. adjust color, cast dtype, rescale (to ensure the longer side does not exceed max size supported, otherwise leave as is)
    # equivalent to prep_im_for_blob in base_dataset.py
    im = img.clone() # important! Don't alter the original inputs vectors

    pixel_means = torch.from_numpy(ups_cfg.network.pixel_means).to(dtype=im.dtype, device=im.device)
    max_size = ups_cfg.test.max_size    # 2048

    # print("img shape", im.shape) # B*C*192*640

    im = im.to(dtype=torch.float32)
    if ups_cfg.network.use_caffe_model:
        im = im * 255 # pixel_means are in 255 format, but monodepth2 input is in [0,1] format
        im -= pixel_means.reshape((-1, 1, 1))
    else:
        im /= 255.0
        im -= torch.tensor([[[0.485, 0.456, 0.406]]])
        im /= torch.tensor([[[0.229, 0.224, 0.225]]])
    im_shape = im.shape
    im_size_min = np.min(im_shape[2:4])
    im_size_max = np.max(im_shape[2:4])
    
    im_scale = float(1)
    # Prevent the biggest axis from being more than max_size
    if im_size_max > max_size:
        im_scale = float(max_size) / float(im_size_max)
        im = torch.nn.functional.interpolate(im, scale_factor=im_scale, mode='bilinear', align_corners=False)
        # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
        #             interpolation=cv2.INTER_LINEAR)

    # 2. transpose the dimension of image from h,w,c to c,h,w

    # 3. prepare im_info
    im_info = np.array([[im.shape[-2],
                        im.shape[-1],
                        im_scale]], np.float32)

    # 4. (data, None)
    panop_in = []
    for ib in range(im.shape[0]):
        data = {"data": im[ib:ib+1], 
                "im_info": im_info}
        panop_in.append((data, None))

    return panop_in

# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

import sys
import os
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, '../UPSNet'))
from upsnet.config.config import config, update_config
import argparse

def parse_args(description='', inputs=None):
    parser = argparse.ArgumentParser(description=description)
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--eval_only', help='if only eval existing results', action='store_true')
    parser.add_argument('--weight_path', help='manually specify model weights', type=str, default='')

    if inputs is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=inputs)
    # update config
    update_config(args.cfg)

    return args

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
import cv2
class PanopVis:
    def __init__(self, num_cls):
        self.colors = np.random.rand(num_cls, 3)

    def paint(self, imgs, panoptic_logits, alpha=0.5, save_path=None, step=None):

        for ib, panoptic_logit in enumerate(panoptic_logits):
            img = imgs[ib:ib+1]
            ## 1. convert logits to classes using argmax
            num_cls_curimg = panoptic_logit.shape[1]
            panoptic_cls = torch.argmax(panoptic_logit, dim=1) # 1*H*W
            ## 2. prepare image for adding colors
            im = img.cpu().numpy()[0]
            im = im.transpose((1,2,0))

            fig = plt.figure(frameon=False)
            fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
            
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')
            fig.add_axes(ax)
            
            ax.imshow(im)
            ## 3. save_path
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)

            ## 4. loop over each class, create a mask, draw the mask
            for ic in range(num_cls_curimg):
                mask_ic = panoptic_cls == ic
                # print("mask_ic before", type(mask_ic))
                mask_ic = mask_ic.cpu().numpy()
                mask_ic = np.uint8(mask_ic.transpose(1,2,0)) #cv2.UMat(mask_ic) # https://stackoverflow.com/questions/54284937/python-typeerror-umat-missing-required-argument-ranges-pos-2
                # print("mask_ic after", type(mask_ic), mask_ic.dtype, mask_ic.shape)
                contour, hier = cv2.findContours(mask_ic, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # https://github.com/facebookresearch/maskrcnn-benchmark/issues/339
                color = self.colors[ic]
                for c in contour:
                    ax.add_patch(
                        Polygon(
                            c.reshape((-1, 2)),
                            fill=True, facecolor=color, edgecolor='w', linewidth=0.5, alpha=0.2
                        )
                    )
            
            ## 5. save or show
            if save_path is None:
                plt.show()
            else:
                fig.savefig(os.path.join(save_path, "{:0>6}_{:0>2}.png".format(step, ib)), dpi=200)