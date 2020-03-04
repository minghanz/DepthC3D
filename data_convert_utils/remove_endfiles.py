### This file is to modify the split files to exclude first and last files in a sequence. 

import os

def get_depth_path(frame_index, data_path, folder, side_map, side):
    f_str = "{:010d}.png".format(frame_index)
    depth_path = os.path.join(
        data_path,
        folder,
        "proj_depth/groundtruth/image_0{}".format(side_map[side]),
        f_str)
    return depth_path

## load file list
file_list_name = 'splits/eigen_zhou_bench_as_val/val_files.txt'
data_path = 'kitti_data'
side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

with open(file_list_name) as f:
    lines = f.readlines()
    valid_lines = []
    for line in lines:
        ## from original line to full path
        items = line.split()
        folder = items[0]
        if len(items) == 3:
            frame_index = int(items[1])
        else:
            frame_index = 0

        if len(items) == 3:
            side = items[2]
        else:
            side = None

        broken_flag = False
        for i in range(-1,2):
            depth_path = get_depth_path(frame_index+i, data_path, folder, side_map, side)
            if not os.path.exists(depth_path):
                broken_flag = True
                break
        if not broken_flag:
            valid_lines.append(line)

with open("val_valid_files.txt", "w") as f:
    f.writelines(valid_lines)

## load the file and the one before and after it
## if all exist, append the original line to new file