import cv2
import numpy as np
import depthai as dai
from camera import Camera
from utils import *
from depth_test import DepthTest


depth_test = DepthTest()
camera = Camera(dai.DeviceInfo())

ROI = (0, 0, 0, 0)
ROI = (135, 64, 224, 220)
while True:
	key = cv2.waitKey(1)

	# QUIT - press the `q` key
	if key == ord('q'):
		break
	
	camera.update()
	if camera.image_frame is None:
		continue
	# Get the point cloud
	ROI_mask = np.zeros_like(camera.depth_frame)
	ROI_mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]] = 1
	depth = camera.depth_frame * ROI_mask
	point_cloud = camera.rgbd_to_point_cloud(depth, cv2.cvtColor(camera.image_frame, cv2.COLOR_RGB2BGR))

	# SELECT ROI - press the `r` key
	if key == ord('r'):
		ROI = cv2.selectROI("image", camera.image_frame)
		print(ROI)

	ROI_mask = np.zeros_like(camera.image_frame)
	ROI_mask[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]] = 1
	img = camera.image_frame*0.2 + (camera.image_frame * ROI_mask)*0.8
	cv2.imshow("image", img.astype(np.uint8))

	# FIT PLANE - press the `f` key
	if key == ord('f'):
		depth_test.fit_plane(point_cloud)
		depth_test.print_tilt()
		depth_test.visualize_plane_fit(point_cloud)

	# TEST DEPTH - press the `t` key
	if key == ord('t'):
		depth_test.measure(point_cloud)