import cv2
import numpy as np
import depthai as dai
from camera import Camera
from alignment_test import AlignmentTest


found, device_info = dai.Device.getAnyAvailableDevice()

if not found:
	print("No device found")
	exit(1)

camera = Camera(device_info, 0, show_video=False)

alignmentTest = AlignmentTest()

min_hue = 110
max_hue = 130

while True:
	key = cv2.waitKey(1)

	camera.update()

	if camera.image_frame is not None and camera.depth_visualization_frame is not None:

		# alignmentTest.add_frame(camera.depth_frame, camera.image_frame)

		depth_mask = alignmentTest.depth_threshold(camera.depth_frame)
		depth_mask3 = np.stack((depth_mask, camera.depth_frame == 0, ~depth_mask), axis=-1)

		image_mask = alignmentTest.image_threshold(camera.image_frame)
		image_mask3 = np.stack((image_mask, np.zeros_like(image_mask), ~image_mask), axis=-1)

		cv2.imshow(camera.image_window_name, camera.image_frame)
		cv2.imshow(camera.depth_window_name, camera.depth_visualization_frame)
		cv2.imshow("depth_mask", (depth_mask3*255).astype(np.uint8))
		cv2.imshow("image_mask", (image_mask3*255).astype(np.uint8))
		cv2.imshow("mono", camera.mono_frame)


	if key == ord('s'):
		roi = cv2.selectROI(camera.image_window_name, camera.image_frame, False, False)
		if(roi[2] > 0 and roi[3] > 0):
			print(roi)
			alignmentTest.set_roi(roi, camera.image_frame, camera.depth_frame)


	if key == ord('c'):
		alignmentTest.add_frame(camera.depth_frame, camera.image_frame)
		print("Added frame to alignment test")

	if key == ord('r'):
		res = alignmentTest.get_results()
		print(f"Alignment test results: {res}")

		alignmentTest.reset()

	if key == ord('q'):
		break
