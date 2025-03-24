import open3d as o3d
import time
import os
import sys

root_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.append(root_dir)

from utils.box_estimator import BoxEstimator 

script_dir = os.path.dirname(os.path.abspath(__file__))

PATH = os.path.join(script_dir, "..", "media/example_pcls/example_4.ply")

if not os.path.exists(PATH):
    print(f"File not found at {PATH}")
    exit(1)

# Read the pointcloud
raw_pcl = o3d.io.read_point_cloud(PATH)

test_box_estimator = BoxEstimator(1.5)

dimensions = test_box_estimator.process_pcl(raw_pcl)
print("Dimensions: ", dimensions)

while True:
    time.sleep(0.02)
    test_box_estimator.vizualise_box()
