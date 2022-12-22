#!/usr/bin/env python3

# Code copied from main depthai repo, depthai_helpers/projector_3d.py

import numpy as np
import open3d as o3d

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.depth_map = None
        self.rgb = None
        self.pcl = o3d.geometry.PointCloud()

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud")
        self.vis.add_geometry(self.pcl)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.vis.add_geometry(origin)
        view_control = self.vis.get_view_control()
        view_control.set_constant_z_far(1000)
        self.isstarted = False

    def rgbd_to_projection(self, depth_map, rgb, downsample = True, remove_noise = False):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=20000, depth_scale=1000.0
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        
        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        if remove_noise:
            pcd = pcd.remove_statistical_outlier(30, 0.1)[0]
        
        self.pcl.points = pcd.points
        self.pcl.colors = pcd.colors
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))
        return self.pcl

    def visualize_pcd(self):
        self.vis.update_geometry(self.pcl)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()
