import numpy as np
import open3d as o3d
from functools import partial

class PointCloudVisualizer():
    def __init__(self):
        self.pcl = None
        # transform from camera to world orientation (Note the absolute position won't be correct)
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("[DepthAI] Open3D integration demo", 960, 540)
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
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))

        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.3)
            # ctr.camera_local_rotate()
            # ctr.camera_local_translate()
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()