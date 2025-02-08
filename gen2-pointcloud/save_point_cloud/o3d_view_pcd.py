"""minimal point cloud viewer using open3d"""
import argparse

import open3d as o3d

parser = argparse.ArgumentParser(description="minimal point cloud viewer using open3d")
parser.add_argument("pcd", help="the .pcd point cloud file")
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.pcd)
print("loaded ", args.pcd)
print("press 'q' to exit")
o3d.visualization.draw_geometries([pcd])
