import os
import numpy as np 
import random

"""
This file is to generate split files for vkitti dataset
"""
scene_names = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
data_root = "/mnt/storage8t/minghanz/Datasets/vKITTI2"

def count_files(scene_name):
    cam_path = os.path.join(data_root, scene_name, "clone/frames/rgb/Camera_0")
    dep_path = os.path.join(data_root, scene_name, "clone/frames/dep/Camera_0")

    n_files = len([name for name in os.listdir(cam_path) if os.path.isfile(os.path.join(cam_path, name))])

    flist = ["{} {} l".format(scene_name, i) for i in range(1,n_files-1)]

    return n_files, flist
######## 
# def create_filelist(scene_name, n_files):
#     flist = ["{} {} l".format(scene_name, i) for i in range(n_files)]

f_list_tt = []
for scene in scene_names:
    n_files, flist = count_files(scene)
    f_list_tt.extend(flist)
    # print(flist)
    print(n_files)

random.shuffle(f_list_tt)
n_train = int(len(f_list_tt)*0.95)
train_list = f_list_tt[:n_train]
eval_list = f_list_tt[n_train:]

if not os.path.exists("splits/vkitti2"):
    os.mkdir("splits/vkitti2")
with open("splits/vkitti2/train_files.txt", "w") as f:
    for item in train_list:
        f.write("{}\n".format(item))

with open("splits/vkitti2/val_files.txt", "w") as f:
    for item in eval_list:
        f.write("{}\n".format(item))
# print(f_list_tt)
