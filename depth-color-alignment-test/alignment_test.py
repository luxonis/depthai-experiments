import cv2
import numpy as np
from typing import Tuple, Optional

class AlignmentTest:

	def __init__(self):
		self.alignments = []
		self.roi: Optional[Tuple[int, int, int, int]] = None

	def add_frame(self, depth_frame, image_frame):
		image_thresh = self.image_threshold(image_frame)
		depth_thresh = self.depth_threshold(depth_frame)

		unknow_pixels = (depth_frame == 0)

		overlap = (image_thresh == depth_thresh) & ~unknow_pixels

		overlap_area = np.count_nonzero(overlap)
		total_area = np.count_nonzero(~unknow_pixels)

		overlap_ratio = overlap_area / total_area

		self.alignments.append(overlap_ratio)

		return overlap_ratio

	def image_threshold(self, image_frame: np.ndarray):
		if self.roi is None: return np.zeros(image_frame.shape[:2], dtype=np.bool)

		hsv = cv2.cvtColor(image_frame, cv2.COLOR_RGB2HSV)
		return ~((hsv[:,:,0] < (self.avg_hue + 10)) & (hsv[:,:,0] > (self.avg_hue - 10)))


	def depth_threshold(self, depth_frame: np.ndarray):
		if self.roi is None: return np.zeros(depth_frame.shape[:2], dtype=np.bool)

		thresh = depth_frame > (self.avg_depth + 50)

		return thresh

	def reset(self):
		self.alignments = []

	def get_results(self):
		if len(self.alignments) == 0: 
			print("No test frames added yet. Press `c` to capture a frame.")
			return None
		avg_alignment = sum(self.alignments) / len(self.alignments)
		return avg_alignment

	def set_roi(self, roi, gray: np.ndarray, depth_frame: np.ndarray):
		self.roi = roi

		hsv = cv2.cvtColor(gray, cv2.COLOR_RGB2HSV)
		hue = hsv[:,:,0]

		# Crop image and depth frame
		hue = hue[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
		depth_frame = depth_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

		self.min_depth = np.min(depth_frame)
		self.max_depth = np.max(depth_frame)
		self.avg_depth = np.mean(depth_frame)

		self.min_hue = np.min(hue)
		self.max_hue = np.max(hue)
		self.avg_hue = np.mean(hue)

		print(f"ROI set: min_depth={self.min_depth}, max_depth={self.max_depth}, avg_depth={self.avg_depth}, min_hue={self.min_hue}, max_hue={self.max_hue}, avg_hue={self.avg_hue}")

