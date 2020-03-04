### This script is to generate file of data list as input to training for BTS (depth prediction). 

import os

mode = "eigen_zhou" #"vkitti" "eigen_zhou"

if mode == "vkitti":
    split_file_path = os.path.join("splits", mode, "test_files.txt")
    file_for_bts_path = os.path.join("splits", mode, "test_files_bts.txt")

    data_root = "/media/sda1/minghanz/datasets/vkitti2"
    side_map = {"l": 0, "r": 1}
elif mode == "eigen_zhou":
    split_file_path = os.path.join("splits", mode, "val_files_samp.txt")
    file_for_bts_path = os.path.join("splits", mode, "val_files_bts.txt")

    data_root = "/media/sda1/minghanz/datasets/kitti/kitti_data"

    side_map = {"l": 2, "r": 3}

with open(split_file_path) as f:
    lines = f.readlines()
    with open(file_for_bts_path, 'w') as g:
        for line in lines:
            scene, seq, cam = line.split()
            if mode == "vkitti":
                line_rgb_path = os.path.join(data_root, scene, "clone", "frames", "rgb", "Camera_{}".format(side_map[cam]), "rgb_{:05d}.jpg".format(int(seq)))
                line_dep_path = os.path.join(data_root, scene, "clone", "frames", "depth", "Camera_{}".format(side_map[cam]), "depth_{:05d}.png".format(int(seq)))
                g.write("{} {} {}\n".format(line_rgb_path, line_dep_path, 725.0087))
            elif mode == "eigen_zhou": # /media/sda1/minghanz/datasets/kitti/kitti_data/2011_09_26/2011_09_26_drive_0101_sync/image_03/data/0000000706.jpg
                date = scene.split("/")[0]
                line_rgb_path = os.path.join(data_root, scene, "image_0{}".format(side_map[cam]), "data", "{:010d}.jpg".format(int(seq)) )
                line_dep_path = "None" # 721.5377
                g.write("{} {} {}\n".format(line_rgb_path, line_dep_path, 721.5377))



