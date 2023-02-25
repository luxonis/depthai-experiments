import time

import open3d as o3d

import box_estimator

PATH_CALIB = 'example_pcls/calibration/calibration.ply'
PATH_BOX = 'example_pcls/calibration/box.ply'

# Read the pointcloud
calib_pcl = o3d.io.read_point_cloud(PATH_CALIB)
box_pcl = o3d.io.read_point_cloud(PATH_BOX)

test_box_estimator = box_estimator.BoxEstimator(1.5)

# test_box_estimator.calibrate(calib_pcl)
dimensions = test_box_estimator.process_pcl(box_pcl)
print("Dimensions: ", dimensions)

while True:
    time.sleep(0.02)
    test_box_estimator.vizualise_box()
