#!/bin/sh

# weight_folder4="/root/repos/monodepth2/tmp/mono_model_KITTI_Lidar/models_Sat Jan 25 16:57:21 2020/weights_19/"
# weight_folder2="/root/repos/monodepth2/tmp/mono_model_KITTI_Lidar/models_Sat Jan 25 21:42:01 2020/weights_19/"
# weight_folder3="/root/repos/monodepth2/tmp/mono_model_KITTI_Lidar/models_Mon Feb  3 00:23:16 2020/weights_19/"
# weight_folder4="/root/repos/monodepth2/tmp/mono_model_KITTI_Lidar/models_Mon Feb  3 00:23:48 2020/weights_19/"

# weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Sun Feb 16 14:40:47 2020/weights_19/"
# weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Sun Feb 16 14:37:42 2020/weights_19/"
weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Tue Feb 18 20:54:10 2020/weights_17/"
# weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Tue Feb 18 20:39:25 2020/weights_16"
# weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Sun Feb  9 01:09:59 2020/weights_19/"
# weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Mon Feb 10 01:08:02 2020/weights_19/"
# weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Tue Feb 25 00:40:29 2020/weights_19/"
# weight_folder4="/media/sda1/minghanz/tmp/mono_model_KITTI_Lidar/models_Tue Feb 25 12:31:18 2020/weights_19/"

# split="eigen_benchmark"
# dataset_val="kitti"
split="vkitti"
dataset_val="vkitti"
for weight_folder in "$weight_folder4" #"$weight_folder1" "$weight_folder2" "$weight_folder3" "$weight_folder4"
do
    # CUDA_LAUNCH_BLOCKING=1 python compare_eval.py \
    python compare_eval.py \
    --load_weights_folder "$weight_folder" --eval_mono --server "sunny" \
    --min_depth 1 --depth_ref_mode --num_workers 3 --num_layers 50 --eval_split "$split" --dataset_val "$dataset_val" --ext_disp_to_eval "/root/repos/bts/pytorch/result_bts_eigen_v2_pytorch_resnet50.npy" --ext_depth
done

# --save_pred_disps
# --disable_median_scaling
#--ext_disp_to_eval "${weight_folder}disps_${split}_split.npy"
# --depth_ref_mode --ref_depth 10
# --min_depth 1 --depth_ref_mode
# --min_depth 2