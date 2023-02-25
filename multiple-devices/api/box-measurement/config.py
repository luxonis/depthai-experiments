import depthai as dai

COLOR = True 		# Use color camera of mono camera

# DEPTH CONFIG
lrcheck  = True   			# Better handling for occlusions
extended = False  			# Closer-in minimum depth, disparity range is doubled
subpixel = True   			# Better accuracy for longer distance, fractional disparity 32-levels
confidence_threshold = 250 	# 0-255, 255 = low confidence, 0 = high confidence
min_range = 100 			# mm from the camera
max_range = 2000			# mm from the camera

# Median filter
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# POINT CLOUD POST-PROCESSING
downsample = True
downsample_voxel_size = 0.005

remove_noise = True
remove_noise_nb_neighbors = 30
remove_noise_std_ratio = 0.5

crop_point_cloud = True
point_cloud_range = {		# m; relative to calibration board
	"x_min": -0.300, "x_max": 0.300,
	"y_min": -0.300, "y_max": 0.300,
	"z_min": -0.300, "z_max": 0.300
}

# CALIBRATION
calibration_data_dir = '../multi-cam-calibration/calibration_data'  # Path to camera extrinsics relative to main.py

# BOX ESTIMATION
min_box_size = 0.002 # m^3
min_box_height = 0.03 # m