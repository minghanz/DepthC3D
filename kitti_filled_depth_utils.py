import os
from PIL import Image
import zipfile
from tqdm import tqdm

if __name__ == "__main__":

    server = "sunny"

    if server == "sunny":
        data_root = "/media/sda1/minghanz/datasets/kitti/kitti_fill_depth"
        zip_name = "/media/sda1/minghanz/datasets/kitti/kitti_fill_depth_chunk0.zip"
        split_file = "/root/repos/monodepth2/splits/eigen_zhou/train_files.txt"
    elif server == "mcity":
        data_root = "/mnt/storage8t/minghanz/Datasets/KITTI_fill_depth"
        zip_name = "/mnt/storage8t/minghanz/Datasets/KITTI_fill_depth_chunk1.zip"
        split_file = "/home/minghanz/monodepth2/splits/eigen_zhou/train_files.txt"
    elif server == "home":
        data_root = "/home/minghanz/Repos/monodepth2/kitti_filled_depth"
        zip_name = "/home/minghanz/Repos/monodepth2/kitti_filled_depth_chunk2.zip"
        split_file = "/home/minghanz/Repos/monodepth2/splits/eigen_zhou/train_files.txt"

    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    with open(split_file) as f:
        lines = f.readlines()

    ################ create zip file
    # n_lines = len(lines)
    # if server == "sunny":
    #     start = 0
    #     end = int(n_lines * 0.5)
    #     lines_trunk = lines[start:end]
    #     start2 = int(n_lines * 0.8)
    #     end2 = int(n_lines * 0.85)
    #     trunk2 = lines[start2:end2]
    #     lines_trunk.extend(trunk2)
    # elif server == "mcity":
    #     start = int(n_lines * 0.5)
    #     end = int(n_lines * 0.8)
    #     lines_trunk = lines[start:end]
    # elif server == "home":
    #     start = int(n_lines * 0.85)
    #     end = n_lines
    #     lines_trunk = lines[start:end]

    # zip_save = zipfile.ZipFile(zip_name, "w")

    # for line in tqdm(lines_trunk):
    #     words = line.split()
    #     folder = words[0]
    #     frame_id = int(words[1])
    #     side = words[2]

    #     for subfolder in ["unfilled_depth_0{}/data".format(side_map[side]), "filled_depth_0{}/data".format(side_map[side])]:
    #         depth_gt_path = os.path.join(
    #             data_root,
    #             folder,
    #             subfolder, 
    #             "{:010d}.png".format(frame_id))
    #         zip_save.write(depth_gt_path, os.path.relpath(depth_gt_path, data_root), compress_type=zipfile.ZIP_DEFLATED)

    # zip_save.close()
    #########################################

    ### Find not processed items in adjacent frames
    not_found_lines = []
    n_nfound_lines = 0
    for line in lines:
        words = line.split()
        folder = words[0]
        frame_id = int(words[1])
        side = words[2]

        for frame_idx in range(frame_id-1, frame_id+2):
            query_line = "{} {} {}\n".format(folder, frame_idx, side)
            if query_line not in lines:
                print(query_line, end="")
                not_found_lines.append(query_line)
                n_nfound_lines += 1
    
    print(n_nfound_lines)
    ###################################################



    ################ find corrupted files
    # corrupted_list = []
    # n_file = 0
    # for root, dirs, files in os.walk(data_root):
    #     if len(files) > 0:
    #         for filename in files:
    #             if filename.split(".")[-1] == "png":
    #                 full_path = os.path.join(root, filename)
    #                 # print(full_path)
    #                 im = Image.open(full_path)
    #                 try:
    #                     im.verify()
    #                     n_file += 1
    #                 except:
    #                     print("Corrupted file:", full_path)
    #                     corrupted_list.append(full_path)

    # print(corrupted_list)
    # print("Total number of good files:", n_file)
    ######################################################