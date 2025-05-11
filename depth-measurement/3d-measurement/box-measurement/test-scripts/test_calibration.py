import open3d as o3d
import time
import os
import sys

root_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
sys.path.append(root_dir)

from utils.box_estimator import BoxEstimator  # noqa: E402

script_dir = os.path.dirname(os.path.abspath(__file__))

PATH_CALIB = os.path.join(
    script_dir, "..", "media/example_pcls/calibration/calibration.ply"
)
PATH_BOX = os.path.join(script_dir, "..", "media/example_pcls/calibration/box.ply")

if not os.path.exists(PATH_CALIB):
    print(f"Calibration file not found at {PATH_CALIB}")
    exit(1)
if not os.path.exists(PATH_BOX):
    print(f"Box file not found at {PATH_BOX}")
    exit(1)

# Read the pointcloud
calib_pcl = o3d.io.read_point_cloud(PATH_CALIB)
box_pcl = o3d.io.read_point_cloud(PATH_BOX)

test_box_estimator = BoxEstimator(1.5)

test_box_estimator.calibrate(calib_pcl)
dimensions = test_box_estimator.process_pcl(box_pcl)
print("Dimensions: ", dimensions)

while True:
    time.sleep(0.02)
    test_box_estimator.vizualise_box()
