import cv2
import numpy as np
import depthai as dai
from camera import Camera
from alignment_test import AlignmentTest


found, device_info = dai.Device.getAnyAvailableDevice()

if not found:
	print("No device found")
	exit(1)

camera = Camera(device_info, 0)

alignmentTest = AlignmentTest()

min_hue = 110
max_hue = 130

while True:
	key = cv2.waitKey(1)

	camera.update()

	if camera.image_frame is not None and camera.depth_visualization_frame is not None:
		# vis = (camera.image_frame * 0.5 + camera.depth_visualization_frame * 0.5).astype(np.uint8)
		# thresh = alignmentTest.image_threshold(camera.image_frame)
		# vis = (np.stack((thresh,)*3, axis=-1) * camera.image_frame).astype(np.uint8)

		hsv = cv2.cvtColor(camera.image_frame, cv2.COLOR_RGB2HSV)
		thresh = (min_hue < hsv[:,:,0]) & (hsv[:,:,0] < max_hue)
		# vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
		vis = camera.image_frame.copy()
		vis[thresh] = [0, 0, 255]
		cv2.imshow(camera.window_name, vis)

	if key == ord('w'):
		min_hue += 1
		print("min_hue", min_hue)

	if key == ord('s'):
		min_hue -= 1
		print("min_hue", min_hue)

	if key == ord('e'):
		max_hue += 1
		print("max_hue", max_hue)

	if key == ord('d'):
		max_hue -= 1
		print("max_hue", max_hue)
		
	if key == ord('c'):
		alignmentTest.add_frame(camera.depth_frame, camera.image_frame)
		print("Added frame to alignment test")

	if key == ord('r'):
		res = alignmentTest.get_results()
		print(f"Alignment test results: {res}")

		alignmentTest.reset()

	if key == ord('q'):
		break
