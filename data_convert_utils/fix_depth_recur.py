## This is to fix the too long path of improved depth images from data_depth_annotated.zip expanded by unzip.py
## From: 
## gt_depth_path = os.path.join(data_path, folder, "proj_depth",
                                        # "groundtruth", "image_02", "{:010d}.png".format(frame_id), 'val' or 'train', folder.split("/")[1], "proj_depth",
                                        # "groundtruth", "image_02", "{:010d}.png".format(frame_id))
## To: 
## gt_depth_path = os.path.join(data_path, folder, "proj_depth",
                                        # "groundtruth", "image_02", "{:010d}.png".format(frame_id))

import os
import glob
import re
import shutil
from tqdm import tqdm

data_root = "/media/sda1/minghanz/datasets/kitti/kitti_data"

path_pattern = re.compile(".+?png")
wanted_files = glob.glob(data_root+"/*/*/proj_depth/**/*.png", recursive=True)
print(len(wanted_files))
for f in tqdm(wanted_files):
    if os.path.isfile(f):
        result = path_pattern.search(f)
        # result = re.search(".+?sync", f)
        wanted_path = result.group(0)
        if wanted_path == f:
            print("Already processed: {}".format(wanted_path))
        else:
            wanted_path_temp = wanted_path + "_"
            shutil.copyfile(f, wanted_path_temp)
            shutil.rmtree(wanted_path)
            # shutil.move
            os.rename(wanted_path_temp, wanted_path)
        # break