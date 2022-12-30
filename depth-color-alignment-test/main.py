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

		image_segmentation, depth_segmentation, error, selection = alignmentTest.update(camera.depth_frame, camera.image_frame)

		image_segmentation_vis = image_segmentation.copy()
		depth_segmentation_vis = depth_segmentation.copy()

		cv2.putText(image_segmentation_vis, "Image segmentation", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
		cv2.putText(depth_segmentation_vis, "Depth segmentation", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
		cv2.putText(selection, "Selection", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

		res = alignmentTest.get_results()
		if res is not None:
			cv2.putText(error, f"Error: {res*100:.2f}%", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
		else:
			cv2.putText(error, "Error: N/A", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

		row1 = np.hstack((camera.image_frame, camera.depth_visualization_frame))
		row2 = np.hstack((image_segmentation_vis, depth_segmentation_vis))
		row3 = np.hstack((selection, error))
		visualization = np.vstack((row1, row2, row3))
		cv2.imshow("Visualization", visualization)

	if key == ord('r'):
		roi = cv2.selectROI(camera.image_window_name, camera.image_frame, False, False)
		if(roi[2] > 0 and roi[3] > 0):
			print(roi)
			alignmentTest.set_roi(roi, camera.image_frame, camera.depth_frame)

	if key == ord('s'):
		cv2.imwrite("image.png", depth_segmentation)

	if key == ord('q'):
		result = "====================\n"
		result += f"Camera {camera.mxid}:\n"
		result += f"Center offset: {np.mean(alignmentTest.center_offsets):.2f} mm\n"
		result += f"Border width: {np.mean(alignmentTest.border_widths):.2f} mm\n"

		with open("results.txt", "a") as f:
			f.write(result)
		break
