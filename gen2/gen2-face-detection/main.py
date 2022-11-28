#!/usr/bin/env python3

from depthai_sdk import Camera

with Camera() as cam:
    rgb_camera = cam.create_camera(out=True)
    face_det = cam.create_nn("face_detection_yunet_160x120", rgb_camera, conf=0.5, out=True)

    cam.start()
    
    vis = cam.create_visualizer(rgb_camera, face_det)

    while cam.running():
        msgs = cam.get_synced_msgs()
        vis.visualize(msgs)

        cam.poll()
