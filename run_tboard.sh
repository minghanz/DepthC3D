#!/bin/sh
# tensorboard --logdir=~/tmp/mono_model_TUM --samples_per_plugin images=20
# tensorboard --logdir=~/tmp/mono_model_KITTI_Lidar --samples_per_plugin images=20
tensorboard --logdir=./tmp/mono_model_KITTI_Lidar --samples_per_plugin images=30 #--port=7007