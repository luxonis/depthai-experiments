import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two rectified images
imgLeft = cv2.imread('rectified_left.png', 0)
imgRight = cv2.imread('rectified_right.png', 0)

# Initialize a StereoBM object
stereo = cv2.StereoBM_create()

# Compute the disparity map
stereo = cv2.StereoBM_create()
# Set the parameters for StereoSGBM
stereo.setBlockSize(9)
stereo.setMinDisparity(0)
stereo.setNumDisparities(64)
stereo.setUniquenessRatio(10)
stereo.setSpeckleWindowSize(0)
stereo.setSpeckleRange(0)
stereo.setDisp12MaxDiff(0)
disparity = stereo.compute(imgLeft, imgRight)

# Normalize the disparity map for visualization
disparityShow = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the rectified frames and the disparity map
cv2.imshow('Left Frame', imgLeft)
cv2.imshow('Right Frame', imgRight)
cv2.imshow('Disparity Map', disparityShow)
cv2.waitKey(0)

# Select a subset of the array to plot
sub_arr = disparity[405:440, 585:614]

# Plot the subset
plt.imshow(sub_arr)
plt.colorbar()
plt.show()

# Compute and plot the distribution of values in the subset
plt.hist(sub_arr.flatten(), bins=50)
plt.title("Distribution of disparity in subset")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show()

# Compute and print the median of the subset
median = np.median(sub_arr)
print(f"Median pixel value: {median}")

dhorScaleFactor = 1597000 / 2
depth = dhorScaleFactor / median
print(f"Depth: {depth} mm")
cv2.destroyAllWindows()