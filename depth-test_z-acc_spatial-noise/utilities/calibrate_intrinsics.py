import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1])) # enable importing from parent directory

import cv2
import numpy as np
import depthai as dai
from oak_camera import OakCamera
from astra_camera import AstraCamera
from utils import *
from openni import openni2
import open3d as o3d

openni2.initialize()
camera = AstraCamera(openni2.Device.open_any())
# camera = OakCamera(dai.DeviceInfo())

checkerboard_inner_size = (9, 6)
square_size = 0.0252 # m

corners_world = np.zeros((1, checkerboard_inner_size[0] * checkerboard_inner_size[1], 3), np.float32)
corners_world[0,:,:2] = np.mgrid[0:checkerboard_inner_size[0], 0:checkerboard_inner_size[1]].T.reshape(-1, 2)
corners_world *= square_size

corners_world_list = []
corners_image_list = []

while True:
	key = cv2.waitKey(1)
	camera.update()

	if key == ord('q'):
		break

	if key == ord('c'):
		print("Finding checkerboard corners ...")
		gray = cv2.cvtColor(camera.image_frame, cv2.COLOR_BGR2GRAY)
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
			camera.image_frame.copy(), checkerboard_inner_size, corners, found
		)

		cv2.imshow(camera.window_name, corners_visualization)
		cv2.waitKey()

		corners_world_list.append(corners_world)
		corners_image_list.append(corners)

		print("Added calibration frame")

	if key == ord('s'):
		print("Saving calibration ...")
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
			corners_world_list, corners_image_list, gray.shape[::-1], None, np.zeros((1, 5))
		)

		print("Camera matrix:")
		print(mtx)
		print("Distortion coefficients:")
		print(dist)

		np.save("astra_intrinsics.npy", mtx)

		break





