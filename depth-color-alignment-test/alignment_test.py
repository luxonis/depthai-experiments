import cv2
import numpy as np

class AlignmentTest:

	def __init__(self):
		self.alignments = []

	def add_frame(self, depth_frame, image_frame):
		image_thresh = self.image_threshold(image_frame)
		depth_thresh = self.depth_threshold(depth_frame)

		unknow_pixels = depth_frame == 0

		overlap = image_thresh & depth_thresh & ~unknow_pixels

		overlap_area = np.count_nonzero(overlap)
		total_area = np.count_nonzero(~unknow_pixels)

		overlap_ratio = overlap_area / total_area

		self.alignments.append(overlap_ratio)

		return overlap_ratio

	def image_threshold(self, image_frame):
		hsv = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)

		lower = np.array([110,50,50])
		upper = np.array([130,255,255])

		thresh = cv2.inRange(hsv, lower, upper)

		return thresh


	def depth_threshold(self, depth_frame):
		# Apply thresholding
		thresh = depth_frame < 1000

		return thresh

	def reset(self):
		self.alignments = []

	def get_results(self):
		avg_alignment = sum(self.alignments) / len(self.alignments)
		return avg_alignment