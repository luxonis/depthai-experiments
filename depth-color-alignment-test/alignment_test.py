import cv2
import numpy as np
from typing import Tuple, Optional
from collections import deque

class AlignmentTest:

	def __init__(self):
		self.alignments = deque(maxlen=20)
		self.roi: Optional[Tuple[int, int, int, int]] = None

	def add_frame(self, depth_frame, image_frame):
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
		
		error_vis = np.stack((error, )*3, axis=-1).astype(np.uint8) * 255
		selection_vis = np.stack((selection, )*3, axis=-1).astype(np.uint8) * 255

		return image_segmentation, depth_segmentation, error_vis, selection_vis

	def image_threshold(self, image_frame: np.ndarray):
		zeros = np.zeros(image_frame.shape[:2], dtype=np.bool)
		if self.roi is None: return zeros, zeros

		hsv = cv2.cvtColor(image_frame, cv2.COLOR_RGB2HSV)
		hue = hsv[:,:,0]

		elevated = (hue < (self.avg_hue + 10)) & (hue > (self.avg_hue - 10))
		floor = ~elevated

		return elevated, floor


	def depth_threshold(self, depth_frame: np.ndarray):
		zeros = np.zeros(depth_frame.shape[:2], dtype=np.bool)
		if self.roi is None: return zeros, zeros

		padding_top = 20 # mm
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
		depth_elevated_count = depth_elevated.shape[0] * depth_elevated.shape[1]
		depth_count = depth_frame.shape[0] * depth_frame.shape[1]

		self.avg_depth_elevated = depth_elevated_sum / depth_elevated_count
		self.avg_depth_floor = (depth_sum - depth_elevated_sum) / (depth_count - depth_elevated_count)


		self.avg_hue = np.mean(hue_elevated)
