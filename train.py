# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions


import sys
import os
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, '../UPSNet'))
from upsnet.config.config import config
from wrap_to_panoptic import parse_args

options = MonodepthOptions()
opts, rest = options.parse()
ups_arg = None
if opts.use_panoptic:
    ups_arg = parse_args(inputs=rest)
    print("ups_arg:", ups_arg)


if __name__ == "__main__":
    trainer = Trainer(opts, ups_arg, config)
    trainer.train()
