import cv2
import numpy as np
import depthai as dai
from camera import Camera
from oak_camera import OakCamera
from astra_camera import AstraCamera
from utils import *
from depth_test import DepthTest
import config
from openni import openni2
import open3d as o3d


openni2.initialize()

depth_test = DepthTest()

oak_camera = OakCamera(dai.DeviceInfo())
astra_camera = AstraCamera(openni2.Device.open_any())

cameras = [oak_camera, astra_camera]

selected_camera = cameras[0]
testing = False
running = True

def quit_callback():
	global running
	running = False

def start_test_callback():
	global testing
	if not depth_test.fitted:
		print("❗WARNING: Plane not fitted, using default values")
	
	print("Testing started ...")
	testing = True

def fit_plane_callback():
	depth_test.fit_plane(selected_camera.point_cloud)
	depth_test.print_tilt()
	# depth_test.visualize_plane_fit(point_cloud)

def select_camera_callback(id: int):
	global selected_camera
	if id < len(cameras) and id >= 0:
		selected_camera = cameras[id]
		print(f"Selected camera: {selected_camera.window_name}")

def save_point_clouds_callback():
	for camera in cameras:
		o3d.io.write_point_cloud(f"point-clouds/{camera.window_name}.ply", camera.point_cloud)

	print("Point clouds saved")

# point cloud visualization window
point_cloud_window = o3d.visualization.VisualizerWithKeyCallback()
point_cloud_window.create_window("Point Cloud")

point_cloud_window.register_key_callback(ord('Q'), lambda vis: quit_callback())
point_cloud_window.register_key_callback(ord('F'), lambda vis: fit_plane_callback())
point_cloud_window.register_key_callback(ord('T'), lambda vis: start_test_callback())
point_cloud_window.register_key_callback(ord('S'), lambda vis: save_point_clouds_callback())
point_cloud_window.register_key_callback(ord('1'), lambda vis: select_camera_callback(0))
point_cloud_window.register_key_callback(ord('2'), lambda vis: select_camera_callback(1))

for camera in cameras:
	point_cloud_window.add_geometry(camera.point_cloud)
point_cloud_window.get_view_control().set_constant_z_far(15)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
point_cloud_window.add_geometry(origin)

while running:
	key = cv2.waitKey(1)

	for camera in cameras:
		camera.update()
		point_cloud_window.update_geometry(camera.point_cloud)

	point_cloud_window.poll_events()
	point_cloud_window.update_renderer()

	if testing:
		depth_test.measure(selected_camera.point_cloud)

		if depth_test.samples >= config.n_samples:
			print()
			testing = False
			depth_test.print_results()
			depth_test.reset()