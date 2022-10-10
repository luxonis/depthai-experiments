import open3d as o3d
from camera import Camera
from typing import List
import numpy as np
import config

class PointCloudVisualizer:
	def __init__(self, cameras: List[Camera]):
		self.cameras = cameras
		self.pointcloud = o3d.geometry.PointCloud()

		self.pointcloud_window = o3d.visualization.VisualizerWithKeyCallback()
		self.pointcloud_window.register_key_callback(ord('C'), lambda vis: self.align_point_clouds())
		self.pointcloud_window.register_key_callback(ord('D'), lambda vis: self.toggle_depth())
		self.pointcloud_window.register_key_callback(ord('S'), lambda vis: self.save_point_cloud())
		self.pointcloud_window.create_window(window_name="Pointcloud")
		self.pointcloud_window.add_geometry(self.pointcloud)
		origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
		self.pointcloud_window.add_geometry(origin)
		view = self.pointcloud_window.get_view_control()
		view.set_constant_z_far(config.max_range*2)

	def update(self):
		self.pointcloud.clear()

		for camera in self.cameras:
			self.pointcloud += camera.point_cloud

		self.pointcloud_window.update_geometry(self.pointcloud)
		self.pointcloud_window.poll_events()
		self.pointcloud_window.update_renderer()

	def align_point_clouds(self):
		print("Aligning point clouds...")

	def toggle_depth(self):
		for camera in self.cameras:
			camera.show_depth = not camera.show_depth

	def save_point_cloud(self):
		for camera in self.cameras:
			o3d.io.write_point_cloud(f"sample_data/pcl_{camera.mxid}.ply", camera.point_cloud)
