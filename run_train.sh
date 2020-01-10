#!/bin/sh
#### Run on KITTI raw dataset using GT lidar points
# python train.py --model_name mono_model_KITTI_Lidar --log_frequency 100 \
# --cvo_loss --batch_size 1 --iters_per_update 6 --scheduler_step_size 10 --supervised_by_gt_depth --cvo_as_loss 

#### train using self-supervised, monitor pose cvo loss usign lidar points
CUDA_LAUNCH_BLOCKING=1 python train.py --model_name mono_model_KITTI_Lidar --log_frequency 100 \
--batch_size 2 --iters_per_update 4 --scheduler_step_size 10  --cvo_loss  --disp_in_loss --min_depth 3 --num_layers 50 --num_workers 6 # --cvo_loss_dense --dense_flat_grid --supervised_by_gt_depth --mask_samp_as_lidar\
#--use_panoptic --cfg /home/minghanz/UPSNet/upsnet/experiments/upsnet_resnet50_cityscapes_1gpu.yaml --weight_path /home/minghanz/UPSNet/model/upsnet_resnet_50_cityscapes_12000.pth
# --sup_cvo_pose_lidar  --multithread
#--cvo_loss_dense \
#--load_weights_folder "/home/minghanz/tmp/mono_model_KITTI_Lidar/models_Sun Dec  1 22:36:45 2019/weights_0" # --num_layers 34#--cvo_loss_dense --ref_depth 10 # --sup_cvo_pose_lidar --supervised_by_gt_depth

# #### Run on TUM dataset using pytorch provided pretrained model
# python train.py --model_name mono_model_TUM --data_path /home/minghanz/Datasets/TUM/rgbd/KITTI_style/ --dataset TUM --split TUM_split --width 384 --height 288 --max_depth 20 --log_frequency 100 \
# --cvo_loss --batch_size 1 --iters_per_update 6 --scheduler_step_size 10 --supervised_by_gt_depth --cvo_as_loss 

# --normalize_inprod_over_pts 
# --cvo_as_loss --load_weights_folder '~/tmp/mono_model_TUM/models_Thu Oct 10 23:29:11 2019/weights_11' 
#--normalize_inprod_over_pts 
#--load_weights_folder '~/tmp/mono_model_TUM/models_Sat Oct 12 00:41:01 2019/weights_19' 


# --data_path /mnt/storage/minghanz_data/TUM/KITTI_style/

### Problem of dataloader killed by signal when launching two program simultaneously should be solved by decrease the number of workders in dataloader (default num_workers = 12).
### It's okay to have one with 12 and one with 6
### See https://github.com/facebookresearch/maskrcnn-benchmark/issues/195. It is probably a problem about memory.
