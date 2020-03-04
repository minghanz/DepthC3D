import numpy as np
from PIL import Image
import pcl
import cv2
import os

## cam intrinsic
# K = np.array([[0.58, 0, 0.5, 0],
#             [0, 1.92, 0.5, 0],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]], dtype=np.float32)
K = np.array([[0.58, 0, 0.5],
            [0, 1.92, 0.5],
            [0, 0, 1]], dtype=np.float32)
img_size = (640, 192)
K[0,:] = K[0,:] * img_size[0]
K[1,:] = K[1,:] * img_size[1]
invK = np.linalg.inv(K)

## cam_pixs

meshgrid = np.meshgrid(range(img_size[0]), range(img_size[1]), indexing='xy')
id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
# print(meshgrid.shape)
print(id_coords.shape)
print(id_coords[:,1,0])
ones = np.ones_like(id_coords[[0]])
id_co_homo = np.concatenate([id_coords, ones], axis=0)
print(id_co_homo.shape)
print(id_co_homo[:,1,0])

id_co_homo_flat = id_co_homo.reshape([3, -1])

id_pts_unit = np.dot(invK, id_co_homo_flat)

## dep and rgb
dep_file_list = [
    "/root/repos/bts/pytorch/result_bts_eigen_v2_pytorch_resnet50kitti/raw/2011_09_30_drive_0028_sync_0000000120.png", 
    "/root/repos/bts/pytorch/result_bts_eigen_v2_pytorch_resnet50kitti/raw/2011_09_30_drive_0028_sync_0000001137.png", 
    "/root/repos/bts/pytorch/result_bts_eigen_v2_pytorch_resnet50kitti/raw/2011_09_30_drive_0028_sync_0000003629.png", 
    "/root/repos/bts/pytorch/result_bts_eigen_v2_pytorch_resnet50kitti/raw/2011_10_03_drive_0034_sync_0000004115.png"
]
rgb_file_list = [
    "2011_09_30/2011_09_30_drive_0028_sync/image_03/data/0000000120.jpg", 
    "2011_09_30/2011_09_30_drive_0028_sync/image_02/data/0000001137.jpg", 
    "2011_09_30/2011_09_30_drive_0028_sync/image_02/data/0000003629.jpg", 
    "2011_10_03/2011_10_03_drive_0034_sync/image_02/data/0000004115.jpg"
]
rgb_root = "/media/sda1/minghanz/datasets/kitti/kitti_data"

for i, dep_file in enumerate(dep_file_list):
    dep = cv2.imread(dep_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    dep = dep.astype(np.float32) / 256
    dep = cv2.resize(dep, img_size)
    dep_flat = dep.reshape(1, -1)

    xyz = dep_flat * id_pts_unit
    print(xyz.shape)

    rgb_file = os.path.join(rgb_root, rgb_file_list[i])
    bgr = cv2.imread(rgb_file)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, img_size)
    
    rgb_flat = rgb.transpose((2,0,1)).reshape((3, -1))

    print(rgb_flat.shape)

    xyzrgb = np.concatenate([xyz, rgb_flat], axis=0)
    xyzrgb = xyzrgb.transpose((1,0))

    cloud = pcl.create_xyzrgb(xyzrgb)
    print(xyzrgb.shape)
    print(xyzrgb.dtype)


    pcd_name = "{}.pcd".format(dep_file.split(".")[0])
    pcl.io.save_pcd(pcd_name, cloud)

    # break

## back projection
## save