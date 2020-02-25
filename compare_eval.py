
### This is to reproduce the depth generation and evaluation in training/validation
### and compare with export_gt_depth.py and evaluate_depth.py to see whether they are consisetent

depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3", "de/si_log", "de/irmse"]

import err_train, err_eval
from trainer import my_collate_fn
import time
import os

import torch
from torch.utils.data import DataLoader

from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import PIL.Image as pil
from kitti_utils import generate_depth_map_original, generate_depth_map, project_lidar_to_img
import numpy as np
import cv2
from layers import depth_to_disp

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
split_file = "test_files.txt"

## GT depth
def gt_depth_from_line(line, opt, data_path, mode="train"):
    folder, frame_id, side = line.split()
    frame_id = int(frame_id)
    if mode == "train":
        im_shape_predefined = [375, 1242]

    if opt.eval_split == "eigen":
        calib_dir = os.path.join(data_path, folder.split("/")[0])
        velo_filename = os.path.join(data_path, folder,
                                        "velodyne_points/data", "{:010d}.bin".format(frame_id))
        if mode == "train":
            velo_rect, P_rect_norm, im_shape  = generate_depth_map(calib_dir, velo_filename, 2)
            gt_depth = project_lidar_to_img(velo_rect, P_rect_norm, im_shape_predefined)      ## ZMH: the way gt is generated I used in training. Resize to a fixed size
            gt_depth = np.expand_dims(gt_depth, 0)
            gt_depth = np.expand_dims(gt_depth, 0)
            gt_depth = torch.from_numpy(gt_depth.astype(np.float32))
        else:
            # gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True) ## ZMH: This won't work because the generate_depth_map function has been redefined.
            gt_depth = generate_depth_map_original(calib_dir, velo_filename, 2, True) ## ZMH: the original function in monodepth2, the size could be different for each img
            # gt_depth = generate_depth_map_original(calib_dir, velo_filename, 2, False) ## ZMH: the original function in monodepth2, use transformed depth

    elif opt.eval_split == "eigen_benchmark":
        # gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
        #                              "groundtruth", "image_02", "{:010d}.png".format(frame_id))
        gt_depth_path = os.path.join(data_path, folder, "proj_depth",
                                        "groundtruth", "image_02", "{:010d}.png".format(frame_id), 'val', folder.split("/")[1], "proj_depth",
                                        "groundtruth", "image_02", "{:010d}.png".format(frame_id))
        if not os.path.exists(gt_depth_path):
            gt_depth_path = os.path.join(data_path, folder, "proj_depth",
                                        "groundtruth", "image_02", "{:010d}.png".format(frame_id), 'train', folder.split("/")[1], "proj_depth",
                                        "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            if not os.path.exists(gt_depth_path):
                raise ValueError("This file does not exist! {} {}".format(folder, frame_id))
        if mode != "train":
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
        else:
            gt_depth = pil.open(gt_depth_path)
            gt_depth = gt_depth.resize(im_shape_predefined[::-1], pil.NEAREST)
            gt_depth = np.array(gt_depth).astype(np.float32) / 256

            gt_depth = np.expand_dims(gt_depth, 0)
            gt_depth = np.expand_dims(gt_depth, 0)
            gt_depth = torch.from_numpy(gt_depth.astype(np.float32)) # input original scale
    
    elif opt.eval_split == "vkitti":
        f_str = "depth_{:05d}.png".format(frame_id)
        gt_depth_path = os.path.join(data_path, folder, "clone", "frames", "depth", "Camera_0", f_str)
        gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        gt_depth = gt_depth.astype(np.float32) / 100
        gt_depth[gt_depth>80] = 0
        height = gt_depth.shape[0]
        gt_depth[:int(height/2)] = 0
        # print(gt_depth.max(), gt_depth.min())
        if mode == "train":
            gt_depth = np.expand_dims(gt_depth, 0)
            gt_depth = np.expand_dims(gt_depth, 0)
            gt_depth = torch.from_numpy(gt_depth.astype(np.float32))
    else:
        raise ValueError("opt.eval_split {} not recognized".format(opt.eval_split))

    return gt_depth

## network -> disp
def network_define(opt, data_path, height, width):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, split_file))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=torch.device("cuda:0"))
        
    if opt.dataset_val[0] == "kitti":
        dataset = datasets.KITTIRAWDataset(data_path, filenames,
                                            height, width,
                                            [0], 4, is_train=False)
    elif opt.dataset_val[0] == "vkitti":
        dataset = datasets.VKITTIDataset(
                    data_path, filenames, height, width,
                    [0], 4, is_train=False)
    # dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
    #                         pin_memory=True, drop_last=False)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False, collate_fn=my_collate_fn) ## the default collate_fn will fail because there are non-deterministic length sample

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device("cuda:0")))

    encoder.cuda(0)
    encoder.eval()
    depth_decoder.cuda(0)
    depth_decoder.eval()

    return encoder, depth_decoder, dataloader, filenames



def main():
    options = MonodepthOptions()
    
    opts, rest = options.parse()

    if opts.server == "mcity":
        datapath_dict = {"kitti": os.path.join(script_path, "kitti_data"),
                        "kitti_odom": None, 
                        "kitti_depth": os.path.join(script_path, "kitti_data"), 
                        "TUM": None, 
                        "lyft_1024": "/mnt/storage8t/minghanz/Datasets/lyft_kitti_seq/train"}
    elif opts.server == "sunny":
        datapath_dict = {"kitti": "/media/sda1/minghanz/datasets/kitti/kitti_data",
                        "kitti_odom": None, 
                        "kitti_depth": "/media/sda1/minghanz/datasets/kitti/kitti_data", 
                        "TUM": None, 
                        "lyft_1024": "/media/sda1/minghanz/datasets/lyft_kitti/train", 
                        "vkitti": "/media/sda1/minghanz/datasets/vkitti2"}
                        #  "lyft_1024": os.path.join(script_path, "data_download/train")} # ZMH: kitti_depth originally not shown as an option here
    elif opts.server == "home":
        datapath_dict = {"kitti": os.path.join(script_path, "kitti_data"),
                        "kitti_odom": None, 
                        "kitti_depth": os.path.join(script_path, "kitti_data"), 
                        "TUM": None, 
                        "lyft_1024": None}
    else:
        raise ValueError("server {} not recognized.".format(opts.server))

    width_dict = {"kitti": 640,
                        "kitti_odom": None, 
                        "kitti_depth": 640, 
                        "TUM": None, 
                        "lyft_1024": 512, 
                        "vkitti": 640} # ZMH: kitti_depth originally not shown as an option here
    height_dict = {"kitti": 192,
                        "kitti_odom": None, 
                        "kitti_depth": 192, 
                        "TUM": None, 
                        "lyft_1024": 224, 
                        "vkitti": 192} # ZMH: kitti_depth originally not shown as an option here # change lyft height from 256 to 192 to 224

    # data_path = datapath_dict["kitti"]
    # width = width_dict["kitti"]
    # height = height_dict["kitti"]
    data_path = datapath_dict[opts.dataset_val[0]]
    width = width_dict[opts.dataset_val[0]]
    height = height_dict[opts.dataset_val[0]]

    if opts.ext_disp_to_eval is None:
        encoder, depth_decoder, dataloader, filenames = network_define(opts, data_path, height, width)

        if opts.save_pred_disps:
            output_path = os.path.join(
                    opts.load_weights_folder, "disps_{}_split.npy".format(opts.eval_split))
            # print("-> Saving predicted disparities to ", output_path)
            disps = []
    else:
        filenames = readlines(os.path.join(splits_dir, opts.eval_split, split_file))
        # Load predictions from file
        print("-> Loading predictions from {}".format(opts.ext_disp_to_eval))
        disps = np.load(opts.ext_disp_to_eval)

    losses_train = {}
    losses_eval = {}
    for item in depth_metric_names:
        losses_train[item] = 0
        losses_eval[item] = 0
    total_n_sp = 0

    if opts.ext_disp_to_eval is None:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input_color = data[("color", 0, 0)].cuda(0)
                output = depth_decoder(encoder(input_color))
                disp = output[("disp", 0)]

                disp_np = disp.cpu().numpy()

                if opts.save_pred_disps:
                    disps.append(disp_np)
                
                line = filenames[i]
                gt_depth_train = gt_depth_from_line(line, opts, data_path, mode="train").cuda(0)
                gt_depth_eval = gt_depth_from_line(line, opts, data_path, mode="eval")

                # ## visualize to check the process is correct, can also be used for qualitative analysis (VKITTI2)
                # if i == 0:
                #     disp_im = (disp_np[0,0,:,:]*255).astype(np.uint8)
                #     # print(disp_np.shape, disp_np.max(), disp_np.min())
                #     img = pil.fromarray(disp_im, mode="L")
                #     img.save(os.path.join(opts.load_weights_folder, "{}.png".format(i)))
                    
                #     gt_disp = depth_to_disp(gt_depth_train, opts.min_depth, opts.max_depth, opts.ref_depth, opts.depth_ref_mode )
                #     disp_np = gt_disp.cpu().numpy()
                #     disp_im = (disp_np[0,0,:,:]*255).astype(np.uint8)
                #     # print(disp_np.shape, disp_np.max(), disp_np.min())
                #     img = pil.fromarray(disp_im, mode="L")
                #     img.save(os.path.join(opts.load_weights_folder, "{}_gt.png".format(i)))

                #     rgb = input_color.cpu().detach().numpy().transpose(0,2,3,1)
                #     rgb = (rgb*255).astype(np.uint8)
                #     rgb_im = pil.fromarray(rgb[0], mode="RGB")
                #     rgb_im.save(os.path.join(opts.load_weights_folder, "{}_rgb.png".format(i)))


                loss_train = err_train.error_disp(disp, gt_depth_train, opts, height, width)
                loss_eval = err_eval.error_disp(disp, gt_depth_eval, opts)

                for item in depth_metric_names:
                    losses_train[item] += loss_train[item]
                    losses_eval[item] += loss_eval[item]
                total_n_sp += 1

                # if i == 10:
                #     break
        if opts.save_pred_disps:
            disps_stack = np.stack(disps)
            np.save(output_path, disps_stack)
            print("-> Saved predicted disparities to ", output_path)
    
    else:
        for i, disp in enumerate(disps):
            # ## visualize to check the process is correct, can also be used for qualitative analysis (VKITTI2)
            # if i == 0:
            #     disp_im = (disp[0,0,:,:]*255).astype(np.uint8)
            #     print(disp_im.shape, disp_im.max(), disp_im.min())
            #     img = pil.fromarray(disp_im, mode="L")
            #     img.save(os.path.join(opts.load_weights_folder, "{}.png".format(i)))

            disp = torch.from_numpy(disp).to(device="cuda:0", dtype=torch.float32)

            line = filenames[i]
            gt_depth_train = gt_depth_from_line(line, opts, data_path, mode="train").cuda(0)
            gt_depth_eval = gt_depth_from_line(line, opts, data_path, mode="eval")

            loss_train = err_train.error_disp(disp, gt_depth_train, opts, height, width)
            loss_eval = err_eval.error_disp(disp, gt_depth_eval, opts)

            for item in depth_metric_names:
                losses_train[item] += loss_train[item]
                losses_eval[item] += loss_eval[item]
            total_n_sp += 1


    for item in depth_metric_names:
        losses_train[item] = losses_train[item] / total_n_sp
        losses_eval[item] = losses_eval[item] / total_n_sp
        print(item, "train:", losses_train[item], ", eval:", losses_eval[item])
    print("total # of samples:", total_n_sp)
        

if __name__ == "__main__":
    time_0 = time.time()
    main()
    time_1 = time.time()
    print("Time elapsed:", time_1-time_0)
    



## disp -> depth

## pred depth, GT depth -> error

