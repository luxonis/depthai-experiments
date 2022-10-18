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
astra_camera = AstraCamera(openni2.Device.open_any())

checkerboard_inner_size = (9, 6)
square_size = 0.0252 # m

corners_world = np.zeros((1, checkerboard_inner_size[0] * checkerboard_inner_size[1], 3), np.float32)
corners_world[0,:,:2] = np.mgrid[0:checkerboard_inner_size[0], 0:checkerboard_inner_size[1]].T.reshape(-1, 2)
corners_world *= square_size

corners_world_list = []
corners_image_list = []

while True:
	key = cv2.waitKey(1)
	astra_camera.update()

	if key == ord('q'):
		break

	if key == ord('c'):
		print("Finding checkerboard corners ...")
		gray = cv2.cvtColor(astra_camera.image_frame, cv2.COLOR_BGR2GRAY)
		found, corners = cv2.findChessboardCorners(
			gray, checkerboard_inner_size, 
			cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
		)

		if not found: 
			print("❗Checkerboard not found")
			continue

		corners = cv2.cornerSubPix(
			gray, corners, (11,11), (-1,-1), 
			(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		)

		corners_visualization = cv2.drawChessboardCorners(
			astra_camera.image_frame.copy(), checkerboard_inner_size, corners, found
		)

		cv2.imshow(astra_camera.window_name, corners_visualization)
		cv2.waitKey()

		corners_world_list.append(corners_world)
		corners_image_list.append(corners)

		print("Added calibration frame")

	if key == ord('s'):
		print("Saving calibration ...")
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
			corners_world_list, corners_image_list, gray.shape[::-1], None, None
		)

		print("Camera matrix:")
		print(mtx)
		print("Distortion coefficients:")
		print(dist)

		break





