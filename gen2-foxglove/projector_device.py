import numpy as np
import open3d as o3d


class PointCloudVisualizer():
    def __init__(self):
        self.pcl = None
        # transform from camera to world orientation (Note the absolute position won't be correct)
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.isstarted = False

    def visualize_pcl(self, pcl_data, downsample=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl_data)
        pcd.remove_non_finite_points()
        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.03)
        # Remove noise
        # pcd = pcd.remove_statistical_outlier(30, 0.1)[0]
        if self.pcl is None:
            self.pcl = pcd
        else:
            self.pcl.points = pcd.points
        # Rotate the pointcloud such that it is in the world coordinate frame (easier to visualize)
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0, 0, 0], dtype=np.float64))
