import pcl
import numpy as np
import torch

# cloud = pcl.PointCloud(np.array([[1,2,3,0]], dtype='f4')) # float32, n*4
# cloud = pcl.create_xyz(array) # n*3

# cloud = pcl.create_xyzrgb(array) # n*6
# cloud.to_ndarray()

# vis = pcl.Visualizer()
# vis.addPointCloud(cloud)
# vis.addCoordinateSystem()

# vis.spin()

def visualize_pcl(xyz, rgb=None, intensity=None):
    """Inputs are tensors of shape B*C*N
    """
    ## 1. tensor to np array
    B = xyz.shape[0]
    xyz_np = xyz.cpu().numpy().swapaxes(1,2)
    if rgb is not None:
        rgb_np = rgb.cpu().numpy().swapaxes(1,2) * 255
        xyz_rgb = np.concatenate((xyz_np, rgb_np), axis=2)
        print("xyz shape", xyz_np.shape)
        print("rgb shape", rgb_np.shape)
        print("xyz_rgb shape", xyz_rgb.shape)
    elif intensity is not None:
        intensity_np = intensity.cpu().numpy().swapaxes(1,2)
        xyz_inten = np.concatenate((xyz_np, intensity_np), axis=2)

    ## 2. np array to pcl cloud objects
    ## 3. create visualize window 
    for ib in range(B):
        if rgb is not None:
            cloud = pcl.create_xyzrgb(xyz_rgb[ib])
            print(cloud.ptype)
        elif intensity is not None:
            cloud = pcl.PointCloud(xyz_inten[ib])
        else:
            cloud = pcl.create_xyz(xyz_np[ib])
        
        print(cloud.to_ndarray())
        vis = pcl.Visualizer()
        vis.addPointCloud(cloud)
        vis.addCoordinateSystem()
        vis.spin()
        pcl.io.save_pcd('{}.pcd'.format(ib), cloud)