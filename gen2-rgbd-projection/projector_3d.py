#!/usr/bin/env python3

# Code copied from main depthai repo, depthai_helpers/projector_3d.py

import numpy as np
import open3d as o3d

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.pcl = None
        # transform from camera to world orientation (Note the absolute position won't be correct)
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False

    def depth_to_projection(self, depth_map, stride=1, downsample=False):
        depth_o3d = o3d.geometry.Image(depth_map)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.pinhole_camera_intrinsic, stride=stride)
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.pinhole_camera_intrinsic, stride=stride)
            # Remove noise
            pcd = pcd.remove_statistical_outlier(30, 0.1)[0]
            self.pcl.points = pcd.points
        # Rotate the pointcloud such that it is in the world coordinate frame (easier to visualize)
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))
        if downsample:
            self.pcl.voxel_down_sample(voxel_size=0.05)
        return self.pcl

    def rgbd_to_projection(self, depth_map, rgb, downsample=False):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False, depth_trunc=6)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            # Remove noise
            pcd = pcd.remove_statistical_outlier(30, 0.1)[0]

            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors

        # Rotate the pointcloud such that it is in the world coordinate frame  (easier to visualize)
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))
        if downsample:
            self.pcl.voxel_down_sample(voxel_size=0.05)
        return self.pcl

    def visualize_pcd(self):
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()
