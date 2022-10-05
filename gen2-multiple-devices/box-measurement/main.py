import cv2
import depthai as dai
from camera import Camera
from typing import List
import config
from box_estimator import BoxEstimator
import open3d as o3d

device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(key=lambda x: x.getMxId(), reverse=True) # sort the cameras by their mxId

cameras: List[Camera] = []

for device_info in device_infos:
    cameras.append(Camera(device_info, len(cameras)+1, show_video=False, show_point_cloud=False))


box_estimator = BoxEstimator(config.max_range/1000)

pointcloud = o3d.geometry.PointCloud()

while True:
    key = cv2.waitKey(1)

    # QUIT - press `q` to quit
    if key == ord('q'):
        break

    # TOGGLE DEPTH VIEW - press `d` to toggle depth view
    if key == ord('d'):
        for camera in cameras:
            camera.show_detph = not camera.show_detph

    pointcloud.clear()
    for camera in cameras:
        camera.update()
        pointcloud += camera.point_cloud


    l, w, h = box_estimator.process_pcl(pointcloud)
    if(l * w * h  > config.min_box_size):
        print(f"Length: {l:.2f}, Width: {w:.2f}, Height:{h:.2f}")
        box_estimator.vizualise_box()

    for camera in cameras:
        img = box_estimator.vizualise_box_2d(camera.intrinsics, camera.world_to_cam, camera.image_frame)
        cv2.imshow(camera.window_name, img)
    
