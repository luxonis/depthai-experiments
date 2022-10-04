import depthai as dai

COLOR = True 		# Use color camera of mono camera

# DEPTH CONFIG
lrcheck  = True   			# Better handling for occlusions
extended = False  			# Closer-in minimum depth, disparity range is doubled
subpixel = True   			# Better accuracy for longer distance, fractional disparity 32-levels
confidence_threshold = 250 	# 0-255, 255 = low confidence, 0 = high confidence
min_range = 100 			# mm
max_range = 2000			# mm

# Median filter
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# CALIBRATION
calibration_data_dir = '../multi-cam-calibration/calibration_data' # Path to camera extrinsics relative to main.py

