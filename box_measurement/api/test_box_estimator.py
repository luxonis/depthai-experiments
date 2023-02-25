import time

import open3d as o3d

import box_estimator

PATH = 'example_pcls/example_5.ply'

# Read the pointcloud
raw_pcl = o3d.io.read_point_cloud(PATH)

test_box_estimator = box_estimator.BoxEstimator(1.5)

dimensions = test_box_estimator.process_pcl(raw_pcl)
print("Dimensions: ", dimensions)

while True:
    time.sleep(0.02)
    test_box_estimator.vizualise_box()
