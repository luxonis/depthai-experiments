import cv2
import shutil
import numpy as np
import depthai as dai
from camera import Camera
from oak_camera import OakCamera
from replay_camera import ReplayCamera
from opencv_replay import OpenCVCamera
from utils import *
from depth_test import DepthTest
import config
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import PointCloud2


from rosbags.rosbag2 import Writer
from rosbags.serde import cdr_to_ros1, serialize_cdr, deserialize_cdr
from rosbags.image import message_to_cvimage
# from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2
# from rosbags.typesys.types import geometry_msgs__msg__PointStamped as PointStampedMsg
# from rosbags.typesys.types import sensor_msgs__msg__Image as Image
# from rosbags.typesys.types import tf2_msgs__msg__TFMessage as tf2_msg
# from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
# from rosbags.typesys.types import std_msgs__msg__Header as Header
from rclpy.time import Time
from rclpy.duration import Duration

from rclpy.node import Node
import rclpy

import std_msgs.msg as std_msgs
# import Header
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PointStamped

from rclpy.serialization import serialize_message

from point_cloud2 import create_cloud_xyzrgb32, create_cloud_xyz32

rclpy.init()
timeout_sec = 0.2
timeout_duration = Duration(seconds=timeout_sec)

roll = -90
pitch = 0
yaw = -90
r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
quat = r.as_quat()

depth_test = DepthTest()
cameras = []

device_info = dai.DeviceInfo()
if config.path is not None:
	if config.use_opencv:
		replay_camera = OpenCVCamera()
	else:
		replay_camera = ReplayCamera(device_info)
	cameras.append(replay_camera)
else:
	oak_camera = OakCamera(device_info)
	cameras.append(oak_camera)


if len(cameras) == 0:
	print("❗ERROR: No cameras found")
	exit()

selected_camera = cameras[0]
testing = False
running = True
solid_color = False

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
		o3d.io.write_point_cloud(f"utilities/{camera.window_name}.ply", camera.point_cloud)

	print("Point clouds saved")

def visualize_plane_callback():
	if depth_test.plane_fit_visualization:
		depth_test.hide_plane_fit_visualization()
	else:
		print("Plane fit visualization")
		print(" - red - original fitted plane")
		print(" - green - corrected fitted plane")
		print(" - colored - corrected point cloud")
		depth_test.show_plane_fit_visualization(selected_camera.point_cloud)

def set_ground_truth_callback():
	distance = depth_test.set_ground_truth(selected_camera.point_cloud)
	print(f"Ground truth set to {distance} m")

def toggle_color_callback():
	global solid_color
	solid_color = not solid_color
# point cloud visualization window
point_cloud_window = o3d.visualization.VisualizerWithKeyCallback()
point_cloud_window.create_window("Point Cloud")

point_cloud_window.register_key_callback(ord('Q'), lambda vis: quit_callback())
point_cloud_window.register_key_callback(ord('F'), lambda vis: fit_plane_callback())
point_cloud_window.register_key_callback(ord('V'), lambda vis: visualize_plane_callback())
point_cloud_window.register_key_callback(ord('T'), lambda vis: start_test_callback())
point_cloud_window.register_key_callback(ord('G'), lambda vis: set_ground_truth_callback())
point_cloud_window.register_key_callback(ord('S'), lambda vis: save_point_clouds_callback())
point_cloud_window.register_key_callback(ord('C'), lambda vis: toggle_color_callback())
point_cloud_window.register_key_callback(ord('1'), lambda vis: select_camera_callback(0))
point_cloud_window.register_key_callback(ord('2'), lambda vis: select_camera_callback(1))

for camera in cameras:
	point_cloud_window.add_geometry(camera.point_cloud)
point_cloud_window.add_geometry(depth_test.plane_fit_pcl)
point_cloud_window.add_geometry(depth_test.plane_fit_corrected_pcl)
point_cloud_window.add_geometry(depth_test.point_cloud_corrected)
point_cloud_window.get_view_control().set_constant_z_far(15)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
point_cloud_window.add_geometry(origin)

node = Node('pointcloud_publisher')
pc_pub = node.create_publisher(PointCloud2, '/point_cloud', 10)
pc_corrected_pub = node.create_publisher(PointCloud2, '/point_cloud_corrected', 10)
gt_cloud_pub = node.create_publisher(PointCloud2, '/gt_cloud', 10)
broadcaster = StaticTransformBroadcaster(node)

isBagWriterUp = False
connections = {}

local_header = std_msgs.Header()
local_header.stamp.sec = 0
local_header.stamp.nanosec = 0
local_header.frame_id = "camera_frame"

z_distance_cm = config.real_depth * 100
x_range = z_distance_cm * 0.5
y_range = z_distance_cm * 0.5
step_size = z_distance_cm * 1j

x, y = np.mgrid[-x_range : x_range : step_size, -y_range : y_range : step_size]
positions = np.vstack([x.ravel(), y.ravel()]).T
z_axis = np.ones((positions.shape[0], 1)) * z_distance_cm
gt_plane = np.hstack([positions, z_axis])
print(f'Original GT shape is -> {gt_plane.shape}')
gt_plane = gt_plane / 100
R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)

while running:
	key = cv2.waitKey(1)

	for camera, color in zip(cameras, [(1,0,0), (0,0,1)]):
		camera.update()
		if solid_color:
			camera.point_cloud.paint_uniform_color(color)
		point_cloud_window.update_geometry(camera.point_cloud)
	point_cloud_window.update_geometry(depth_test.plane_fit_pcl)
	point_cloud_window.update_geometry(depth_test.plane_fit_corrected_pcl)
	point_cloud_window.update_geometry(depth_test.point_cloud_corrected)

	point_cloud_window.poll_events()
	point_cloud_window.update_renderer()


	t_stamped = TransformStamped()
	t_stamped.header.stamp = node.get_clock().now().to_msg()
	t_stamped.header.frame_id = "map"
	t_stamped.child_frame_id = "camera_frame"
	t_stamped.transform.translation.x = 0.0
	t_stamped.transform.translation.y = 0.0
	t_stamped.transform.translation.z = 0.0
	t_stamped.transform.rotation.x = quat[0]
	t_stamped.transform.rotation.y = quat[1]
	t_stamped.transform.rotation.z = quat[2]
	t_stamped.transform.rotation.w = quat[3]
	broadcaster.sendTransform(t_stamped)

	org_pointCloud = selected_camera.point_cloud.rotate(R_camera_to_world, center=(0, 0, 0))

	filtered_pts = np.asarray(org_pointCloud.points)
	colors = np.asarray(org_pointCloud.colors)
	cloud = np.concatenate([filtered_pts, colors], axis=-1)


	local_header.stamp = node.get_clock().now().to_msg()
	local_header.frame_id = "camera_frame"

	pc2 = create_cloud_xyzrgb32(local_header, cloud)
	pc_pub.publish(pc2)
	pc2 = create_cloud_xyz32(local_header, gt_plane)
	gt_cloud_pub.publish(pc2)

	corrected_pointCloud = depth_test.point_cloud_corrected.rotate(R_camera_to_world, center=(0, 0, 0))
	filtered_pts = np.asarray(corrected_pointCloud.points)
	colors = np.asarray(corrected_pointCloud.colors)
	corrected_cloud = np.concatenate([filtered_pts, colors], axis=-1)
	pc2 = create_cloud_xyzrgb32(local_header, corrected_cloud)
	pc_corrected_pub.publish(pc2)

	rclpy.spin_once(node, timeout_sec=timeout_sec)

	if testing:
		selected_camera.point_cloud.rotate(R_camera_to_world, center=(0, 0, 0))
		depth_test.point_cloud_corrected.rotate(R_camera_to_world, center=(0, 0, 0))
		depth_test.measure(selected_camera)
		custom_timestamp = Time.from_msg(local_header.stamp).nanoseconds

		if depth_test.samples >= config.n_samples:
			print()
			testing = False
			depth_test.print_results()
			depth_test.reset()
			# writer.close()
			print('Closing the writer!!')

	if testing and 0:
		depth_test.measure(selected_camera)
		custom_timestamp = Time.from_msg(local_header.stamp).nanoseconds
		if not isBagWriterUp:
			isBagWriterUp = True
			if config.output_path.exists():
			    shutil.rmtree(config.output_path)

			writer = Writer(str(config.output_path))
			writer.open()

			connections['/tf_static'] = writer.add_connection('/tf_static', tf2_msg.__msgtype__)
			connections['cloud'] = writer.add_connection('/cloud', PointCloud2.__msgtype__)
			z = config.real_depth

			tf_msg = TFMessage()
			t_stamped = TransformStamped()
			t_stamped.header = local_header
			t_stamped.header.frame_id = "map"
			t_stamped.child_frame_id = "camera_frame"
			t_stamped.transform.translation.x = 0.0
			t_stamped.transform.translation.y = 0.0
			t_stamped.transform.translation.z = 0.0
			t_stamped.transform.rotation.x = quat[0]
			t_stamped.transform.rotation.y = quat[1]
			t_stamped.transform.rotation.z = quat[2]
			t_stamped.transform.rotation.w = quat[3]
			tf_msg.transforms.append(t_stamped)
			ser_tf_msg = serialize_message(tf_msg)

			writer.write(connections['/tf_static'], custom_timestamp, ser_tf_msg)

		filtered_pts = np.asarray(selected_camera.point_cloud.points)
		colors = np.asarray(selected_camera.point_cloud.colors)
		cloud = np.concatenate([filtered_pts, colors], axis=-1)
		local_header.frame_id = "camera_frame"
		pc2 = create_cloud_xyzrgb32(local_header, cloud)
		ser_pc_msg = serialize_message(pc2)
		writer.write(connections['cloud'], custom_timestamp, ser_pc_msg)
		local_header.stamp.sec += 1

		if depth_test.samples >= config.n_samples:
			print()
			testing = False
			depth_test.print_results()
			depth_test.reset()
			writer.close()
			print('Closing the writer!!')
