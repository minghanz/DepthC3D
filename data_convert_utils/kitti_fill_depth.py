
### load file list from split file
### loop over each item, convert self and adjacent depth
### target is a depth image 

from kitti_utils import generate_depth_map, project_lidar_to_img
import os
import numpy as np
from PIL import Image
from depth_filler import fill_depth_colorization
import time
from multiprocessing import Pool, Value

def prep_path_and_filename(data_root, folder, frame_id, subfolder, depth):
    depth_gt_path = os.path.join(
        data_root,
        folder,
        subfolder)
    if not os.path.exists(depth_gt_path):
        os.makedirs(depth_gt_path)

    depth_gt_filename = os.path.join(
        depth_gt_path, 
        "{:010d}.png".format(frame_id))

    save_depth_img(depth, depth_gt_filename)

def save_depth_img(depth_array, filename):
    """using kitti style (uint16 with value=depth*256)"""
    depth_imgarray = (depth_array*256).astype(np.uint16)
    depth_img = Image.fromarray(depth_imgarray, mode="I;16")
    depth_img.save(filename)

def check_existence(data_root, folder, frame_id, subfolder):
    depth_gt_path = os.path.join(
        data_root,
        folder,
        subfolder)

    depth_gt_filename = os.path.join(
        depth_gt_path, 
        "{:010d}.png".format(frame_id))
    
    if not os.path.exists(depth_gt_filename):
        return False
    else:
        img = Image.open(depth_gt_filename)
        try:
            img.verify()
            return True
        except Exception:
            print("Exist but corrupted:", depth_gt_filename)
            return False

class ProcessDepth:
    def __init__(self, server):

        if server == "sunny":
            self.data_path = "/media/sda1/minghanz/datasets/kitti/kitti_data"
            self.ext_data_path = "/media/sda1/minghanz/datasets/kitti/kitti_fill_depth"
        elif server == "mcity":
            self.data_path = "/home/minghanz/monodepth2/kitti_data"
            self.ext_data_path = "/mnt/storage8t/minghanz/Datasets/KITTI_fill_depth"
        elif server == "home":
            self.data_path = "/home/minghanz/Repos/monodepth2/kitti_data"
            self.ext_data_path = "/home/minghanz/Repos/monodepth2/kitti_filled_depth"

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.full_res_shape = (1242, 375)
    
    def __call__(self, line):
        words = line.split()
        folder = words[0]
        frame_id = int(words[1])
        side = words[2]

        # print(line)
        # for frame_id in range(frame_idx-1, frame_idx+2):

        calib_path = os.path.join(self.data_path, folder.split("/")[0])
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(frame_id))

        if not os.path.exists(velo_filename):
            print("Not found:", velo_filename)
            return

        global counter
        with counter.get_lock():
            valid_1 = check_existence(self.ext_data_path, folder, frame_id, "unfilled_depth_0{}/data".format(self.side_map[side]))
            valid_2 = check_existence(self.ext_data_path, folder, frame_id, "filled_depth_0{}/data".format(self.side_map[side]))
            if valid_1 and valid_2:
                counter.value += 1
                i = counter.value
                print(i, "skipped in {} s.".format(time.time()-start_time))
                print(folder, frame_id, side)
                return

        velo_rect, P_rect_norm, im_shape  = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = project_lidar_to_img(velo_rect, P_rect_norm, self.full_res_shape[::-1])   # H*W, raw depth

        # prep_path_and_filename(self.data_path, folder, frame_id, "unfilled_depth_0{}/data".format(self.side_map[side]), depth_gt)

        prep_path_and_filename(self.ext_data_path, folder, frame_id, "unfilled_depth_0{}/data".format(self.side_map[side]), depth_gt)

        ### fill depth
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data/{:010d}.jpg".format(self.side_map[side], frame_id))
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                rgb_img = img.convert('RGB')
        
        rgb_img = rgb_img.resize(self.full_res_shape)
        rgb_imgarray = np.array(rgb_img).astype(np.float32)/255 # H*W*3, between 0~1
        # print(rgb_imgarray.min(), rgb_imgarray.max(), rgb_imgarray.shape, np.median(rgb_imgarray))
        
        depth_filled = fill_depth_colorization(imgRgb=rgb_imgarray, imgDepthInput=depth_gt, alpha=1)

        # prep_path_and_filename(self.data_path, folder, frame_id, "filled_depth_0{}/data".format(self.side_map[side]), depth_filled)

        prep_path_and_filename(self.ext_data_path, folder, frame_id, "filled_depth_0{}/data".format(self.side_map[side]), depth_filled)
    
        # += operation is not atomic, so we need to get a lock:
        with counter.get_lock():
            counter.value += 1
            i = counter.value
            if (i < 10) or (i % 10 == 0 and i < 100) or (i % 100 == 0 and i < 1000) or (i % 1000 == 0 and i < 10000) or (i % 5000 == 0):
                print(i, "finished in {} s.".format(time.time()-start_time))
                print(folder, frame_id, side)

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args

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


    with open(split_file) as f:
        lines = f.readlines()
    
    n_lines = len(lines)
    if server == "sunny":
        start = 0
        end = int(n_lines * 0.5)
        lines_trunk = lines[start:end]
        start2 = int(n_lines * 0.8)
        end2 = int(n_lines * 0.85)
        trunk2 = lines[start2:end2]
        lines_trunk.extend(trunk2)
    elif server == "mcity":
        start = int(n_lines * 0.5)
        end = int(n_lines * 0.8)
        lines_trunk = lines[start:end]
    elif server == "home":
        start = int(n_lines * 0.85)
        end = n_lines
        lines_trunk = lines[start:end]

    # not_found_lines = []
    # n_nfound_lines = 0
    # for line in lines:
    #     words = line.split()
    #     folder = words[0]
    #     frame_id = int(words[1])
    #     side = words[2]

    #     for frame_idx in range(frame_id-1, frame_id+2):
    #         query_line = "{} {} {}\n".format(folder, frame_idx, side)
    #         if query_line not in lines:
    #             # print(query_line, end="")
    #             not_found_lines.append(query_line)
    #             n_nfound_lines += 1
    
    # print(n_nfound_lines)


    DepthFiller = ProcessDepth(server)
    print("os.cpu_count()", os.cpu_count()) # 48

    counter = Value('i', 0)

    start_time = time.time()

    if server == "sunny":
        n_proc = 16
    elif server == "mcity":
        n_proc = 8
    elif server == "home":
        n_proc = 4
    
    with Pool(processes=n_proc, initializer = init, initargs = (counter, )) as pool:
        pool.map(DepthFiller, lines_trunk)
    
    print(counter.value)
    print(time.time() - start_time)

# for i, line in enumerate(lines):
    

        # break

    # if (i % 100 == 0 and i < 1000) or (i % 1000 == 0 and i < 10000) or (i % 5000 == 0):
    #     print("{} finished. {}s passed.".format(i, time.time()-start_time))
    #     print(folder, frame_idx, side)
    
    # break