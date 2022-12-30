import cv2
import numpy as np
from typing import Tuple, Optional
from collections import deque
import config

class AlignmentTest:

	def __init__(self):
		self.alignments = deque(maxlen=20)
		self.border_widths = deque(maxlen=20)
		self.center_offsets = deque(maxlen=20)
		self.roi: Optional[Tuple[int, int, int, int]] = None

	def fit_rect(self, bin_img):
		closing = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
		contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		if len(contours) == 0:
			return None
		cnt = np.vstack(contours).squeeze()
		if len(cnt) < 4:
			return None
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		return box

	def update(self, depth_frame, image_frame):
		image_elevated, image_floor = self.image_threshold(image_frame)
		depth_elevated, depth_floor = self.depth_threshold(depth_frame)


		error = (image_elevated != depth_elevated) & (image_floor != depth_floor)
		selection = (depth_elevated & image_elevated) | (depth_floor & image_floor)

		error_area = np.count_nonzero(error)
		total_area = np.count_nonzero(selection)

		if total_area != 0:
			error_ratio = error_area / total_area
			self.alignments.append(error_ratio)

		# visualization
		image_segmentation = np.stack((image_floor, np.zeros_like(image_elevated), image_elevated), axis=-1).astype(np.uint8) * 255
		depth_segmentation = np.stack((depth_floor, np.zeros_like(depth_elevated), depth_elevated), axis=-1).astype(np.uint8) * 255

		image_box = self.fit_rect(image_elevated.astype(np.uint8)*255)
		depth_box = self.fit_rect(depth_elevated.astype(np.uint8)*255)

		error_vis = np.stack((error, )*3, axis=-1).astype(np.uint8) * 255
		selection_vis = np.stack((selection, )*3, axis=-1).astype(np.uint8) * 255


		if image_box is not None and depth_box is not None:
			cv2.drawContours(image_segmentation, [image_box], 0, (0,255,0), 2)
			cv2.drawContours(depth_segmentation, [depth_box], 0, (0,255,0), 2)


			# calculate border width
			rectangle_perimeter_px = cv2.arcLength(image_box, True)
			px_to_mm = config.rectangle_perimeter_mm / rectangle_perimeter_px
			d_px = error_area / rectangle_perimeter_px
			d_mm = d_px * px_to_mm
			self.border_widths.append(d_mm)

			# calculate center offset
			d = np.linalg.norm(image_box.mean(axis=0) - depth_box.mean(axis=0))*px_to_mm
			self.center_offsets.append(d)

		
		print(f"Center offset: {np.mean(self.center_offsets):.2f} mm")
		print(f"Border width: {np.mean(self.border_widths):.2f} mm")
		

		return image_segmentation, depth_segmentation, error_vis, selection_vis

	def image_threshold(self, image_frame: np.ndarray):
		zeros = np.zeros(image_frame.shape[:2], dtype=np.bool_)
		if self.roi is None: return zeros, zeros

		hsv = cv2.cvtColor(image_frame, cv2.COLOR_RGB2HSV)
		hue = hsv[:,:,0]

		elevated = (hue < (self.avg_hue + 10)) & (hue > (self.avg_hue - 10))
		floor = ~elevated

		return elevated, floor


	def depth_threshold(self, depth_frame: np.ndarray):
		zeros = np.zeros(depth_frame.shape[:2], dtype=np.bool_)
		if self.roi is None: return zeros, zeros

		padding_top = 100 # mm
		padding_bottom = 100 # mm
		mid_depth = (self.avg_depth_elevated + self.avg_depth_floor) / 2
		elevated = (mid_depth > depth_frame) & (depth_frame > (self.avg_depth_elevated - padding_top))
		floor = ((self.avg_depth_floor + padding_bottom) > depth_frame) & (depth_frame > mid_depth)

		return elevated, floor

	def reset(self):
		self.alignments = []

	def get_results(self):
		if len(self.alignments) == 0: 
			print("No test frames added yet. Press `c` to capture a frame.")
			return None
		avg_alignment = sum(self.alignments) / len(self.alignments)
		return avg_alignment

	def set_roi(self, roi, image_frame: np.ndarray, depth_frame: np.ndarray):
		self.roi = roi

		hsv = cv2.cvtColor(image_frame, cv2.COLOR_RGB2HSV)
		hue = hsv[:,:,0]

		# Crop image and depth frame
		hue_elevated = hue[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
		depth_elevated = depth_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

		depth_elevated_sum = np.sum(depth_elevated)
		depth_sum = np.sum(depth_frame)
		depth_elevated_count = depth_elevated.shape[0] * depth_elevated.shape[1] - np.count_nonzero(depth_elevated == 0)
		depth_count = depth_frame.shape[0] * depth_frame.shape[1] - np.count_nonzero(depth_frame == 0)

		self.avg_depth_elevated = depth_elevated_sum / depth_elevated_count
		self.avg_depth_floor = (depth_sum - depth_elevated_sum) / (depth_count - depth_elevated_count)


		self.avg_hue = np.mean(hue_elevated)
