import cv2
import numpy as np
import depthai as dai
from oak_camera import OakCamera
from astra_camera import AstraCamera
from utils import *
import config
from openni import openni2
import open3d as o3d


openni2.initialize()
astra_camera = AstraCamera(openni2.Device.open_any())
oak_camera = OakCamera(dai.DeviceInfo())

# visualization window
point_cloud_window = o3d.visualization.Visualizer()
point_cloud_window.create_window("Point Cloud")
point_cloud_window.add_geometry(oak_camera.point_cloud)
point_cloud_window.add_geometry(astra_camera.point_cloud)
point_cloud_window.get_view_control().set_constant_z_far(15)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
point_cloud_window.add_geometry(origin)


while True:
    astra_camera.update()
    astra_camera.rgbd_to_point_cloud(astra_camera.depth_frame, cv2.cvtColor(astra_camera.image_frame, cv2.COLOR_RGB2BGR))
    oak_camera.update()
    oak_camera.rgbd_to_point_cloud(oak_camera.depth_frame, cv2.cvtColor(oak_camera.image_frame, cv2.COLOR_RGB2BGR))

    point_cloud_window.update_geometry(astra_camera.point_cloud)
    point_cloud_window.update_geometry(oak_camera.point_cloud)
    point_cloud_window.poll_events()
    point_cloud_window.update_renderer()