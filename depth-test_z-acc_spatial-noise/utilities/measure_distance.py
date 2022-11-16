# MEASURE DISTANCE
#
# This script can be used to measure distance between two points in the point cloud
#   - change the path to the point cloud file on line 13
#   - run the script
#   - pick two points in the point cloud
#   - the distance between the two points will be printed
#

import open3d as o3d
import numpy as np

pcl = o3d.io.read_point_cloud("point-clouds/Camera_OAK.ply")

def pick_points(pcd):
    print("")
    print("1) Please pick two points using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

points_ids = pick_points(pcl)

points = np.asarray(pcl.points)[points_ids]

print(points)
print(f"Distance: {np.linalg.norm(points[0] - points[1])}")