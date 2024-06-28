import cv2
import numpy as np
import depthai as dai
from camera import Camera
from oak_camera import OakCamera
from astra_camera import AstraCamera
from utils import *
from depth_test import DepthTest
import config
import open3d as o3d
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation


if config.astra_gt:
	from openni import openni2
	openni2.initialize()

depth_test = DepthTest()
cameras = []
lap_mes=[]
lap_grid=[]
dis_me=[]
std_lap_mes=[]
std_distance_me=[]

try:
	found, device_info = dai.Device.getFirstAvailableDevice()
	oak_camera = OakCamera(device_info)
	cameras.append(oak_camera)
except:
	print("❗WARNING: OAK-D not found")

if config.astra_gt:
	try:
		device_info = openni2.Device.open_any()
		astra_camera = AstraCamera(device_info)
		cameras.append(astra_camera)
	except:
		print("❗WARNING: Astra not found")

if len(cameras) == 0:
	print("❗ERROR: No cameras found")
	exit()

selected_camera = cameras[0]
testing = False
measure = False
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

def set_ground_truth_callback_measurements():
	global measure
	measure=True
	print("Start measurement")
	return measure

def stop_ground_truth_callback_measurements():
	global measure
	measure=False
	print("Stop measurement")
	return measure

def toggle_color_callback():
	global solid_color
	solid_color = not solid_color


def save_results_callback():
	name = selected_camera.mxid if hasattr(selected_camera, "mxid") else "unknown"
	depth_test.save_results(name)



# point cloud visualization window
point_cloud_window = o3d.visualization.VisualizerWithKeyCallback()
point_cloud_window.create_window("Point Cloud")

point_cloud_window.register_key_callback(ord('Q'), lambda vis: quit_callback())
point_cloud_window.register_key_callback(ord('F'), lambda vis: fit_plane_callback())
point_cloud_window.register_key_callback(ord('V'), lambda vis: visualize_plane_callback())
point_cloud_window.register_key_callback(ord('T'), lambda vis: start_test_callback())
point_cloud_window.register_key_callback(ord('G'), lambda vis: set_ground_truth_callback())
point_cloud_window.register_key_callback(ord('P'), lambda vis: save_point_clouds_callback())
point_cloud_window.register_key_callback(ord('C'), lambda vis: toggle_color_callback())
point_cloud_window.register_key_callback(ord('1'), lambda vis: select_camera_callback(0))
point_cloud_window.register_key_callback(ord('2'), lambda vis: select_camera_callback(1))
point_cloud_window.register_key_callback(ord('M'), lambda vis: set_ground_truth_callback_measurements())
point_cloud_window.register_key_callback(ord('S'), lambda vis: stop_ground_truth_callback_measurements())
for camera in cameras:
	point_cloud_window.add_geometry(camera.point_cloud)
point_cloud_window.add_geometry(depth_test.plane_fit_pcl)
point_cloud_window.add_geometry(depth_test.plane_fit_corrected_pcl)
point_cloud_window.add_geometry(depth_test.point_cloud_corrected)
point_cloud_window.get_view_control().set_constant_z_far(15)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
point_cloud_window.add_geometry(origin)

def _WarpImage(img, pnts, size):
	src_pnts = np.array([
		pnts['0'], pnts['1'],
		pnts['2'], pnts['3']
	], dtype=np.float32)
	dst_pnt = np.array([
		[0, 0], [size[0], 0],
		[0, size[1]], [size[0], size[1]]
	], dtype=np.float32)
	warp_mat = cv2.getPerspectiveTransform(src_pnts, dst_pnt)
	img = cv2.warpPerspective(img, warp_mat, size)
	return img
index=0
pic_taken=0
pic_max=10
lap_mes_taken=[]
lap_grid_taken=[]
dis_me_taken=[]
while running:
	key = cv2.waitKey(1)
	#cv2.imshow("Reference picture ", img_ref)
	for camera, color in zip(cameras, [(1,0,0), (0,0,1)]):
		img_ref = cv2.imread("C:\\Users\\matic\\Downloads\\test_target-1.jpg",0)
		camera.update()
		img=camera.image_frame
		lap_img=img
		if len(img) == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		arruco_success=False
		try:
			#aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
			aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
			corners, ids, ret = cv2.aruco.detectMarkers(img, aruco_dict)
			center_dict = {}
			for e, id in enumerate(ids):
				corners_block = corners[e][0]
				center = np.mean(corners_block, axis=0)
				center_dict[str(id[0])] = center
			size=(1200, 800)
			img=_WarpImage(img, center_dict, size)

			corners, ids, ret = cv2.aruco.detectMarkers(img_ref, aruco_dict)
			center_dict = {}
			for e, id in enumerate(ids):
				corners_block = corners[e][0]
				center = np.mean(corners_block, axis=0)
				center_dict[str(id[0])] = center
			size=(1200, 800)
			img_ref=_WarpImage(img_ref, center_dict, size)
			arruco_success=True
		except: 
			if measure:
				print("Face camera to target.")
		if arruco_success:
			img= cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			lap_img =cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=3, borderType=cv2.BORDER_DEFAULT)
			img_ref= cv2.normalize(img_ref, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			lap_img_ref =cv2.Laplacian(img_ref, ddepth=cv2.CV_32F, ksize=3, borderType=cv2.BORDER_DEFAULT)

		cv2.imshow("Laplacian of an image. ", lap_img)


		if solid_color:
			camera.point_cloud.paint_uniform_color(color)
		point_cloud_window.update_geometry(camera.point_cloud)
	point_cloud_window.update_geometry(depth_test.plane_fit_pcl)
	point_cloud_window.update_geometry(depth_test.plane_fit_corrected_pcl)
	point_cloud_window.update_geometry(depth_test.point_cloud_corrected)

	point_cloud_window.poll_events()
	point_cloud_window.update_renderer()
	if measure:
		distance = depth_test.set_ground_truth(selected_camera.point_cloud)
		if arruco_success:
			pic_taken+=1
			lap_grid_taken.append(lap_img)
			lap_mes_taken.append(lap_img.var()/lap_img_ref.var())
			dis_me_taken.append(distance)
			print(pic_taken)
			if pic_taken==pic_max:
				print(lap_img.var())
				print(lap_img_ref.var())
				pic_taken=0
				#plt.plot(dis_me_taken, lap_mes_taken, label=f"$d_m$:{np.mean(dis_me_taken)},d_s: {np.std(dis_me_taken)} $")
				#plt.grid()
				#plt.legend()
				#plt.xlabel("Distance [m]")
				#plt.ylabel("Laplace variance")
				#plt.show()
				#y = np.arange(len(lap_img))
				#x = np.arange(len(lap_img[0]))
				#X, Y=np.meshgrid(x,y)
				#fig = plt.figure()
				#ax = plt.axes(projection='3d')
				#ax.contour3D(X, Y, lap_img, color='viridis')
				#ax.set_xlabel('x')
				#ax.set_ylabel('y')
				#ax.set_zlabel('z')
				#plt.show()
				dis_me.append(np.mean(dis_me_taken))
				lap_mes.append(np.mean(lap_mes_taken))
				lap_grid.append(np.mean(lap_grid_taken, axis=0))
				std_lap_mes.append(np.std(lap_mes_taken))
				std_distance_me.append(np.std(dis_me_taken))
				lap_mes_taken=[]
				lap_grid_taken=[]
				dis_me_taken=[]
				key = cv2.waitKey(0)

	else:
		if len(lap_mes)!=0:
			print(std_lap_mes, std_distance_me)
			plt.errorbar(dis_me,lap_mes, xerr=std_distance_me, yerr=std_lap_mes, fmt="-x")
			plt.grid()
			plt.xlabel("Distance [m]")
			plt.ylabel("Laplace variance")
			plt.show()
			lap_mes=[]
			dis_me=[]
			std_lap_mes=[]
			std_distance_me=[]
	if testing:
		depth_test.measure(selected_camera)

		if depth_test.samples >= config.n_samples:
			print()
			testing = False
			depth_test.print_results()
			save_results_callback()
			depth_test.reset()