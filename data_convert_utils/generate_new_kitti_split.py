### This is to generate a split file for a video sequence, for generating a qualitative video.

import os
import numpy as np 
import sys
script_path = os.path.dirname(__file__)

data_root = "/media/sda1/minghanz/datasets/kitti/kitti_data"
seq = "2011_09_26/2011_09_26_drive_0093_sync"
img_folder = "image_02"
lidar_folder = "velodyne_points"

data_path = os.path.join(data_root, seq, img_folder, "data")
files = os.listdir(data_path)
frame_id_list = [int(f.split(".")[0]) for f in files if os.path.isfile(os.path.join(data_path,f))]
# for f in files:
#     frame_id = int(f.split(".")[0])
#     frame_id_list.append(frame_id)

frame_id_list = sorted(frame_id_list)
lines = ["{} {} l\n".format(seq, frame_id) for frame_id in frame_id_list]

split_file_name = os.path.join(script_path, "../splits/kitti_video/val_files.txt")
with open(split_file_name, "w") as f:
    f.writelines(lines)