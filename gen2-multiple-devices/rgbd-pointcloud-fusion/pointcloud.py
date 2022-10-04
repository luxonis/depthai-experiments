import open3d as o3d
from camera import Camera
from typing import List
import numpy as np
import config

class Pointcloud:
	def __init__(self, cameras: List[Camera]):
		self.cameras = cameras
		self.pointcloud = o3d.geometry.PointCloud()

		self.pointcloud_window = o3d.visualization.Visualizer()
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