#!/usr/bin/env python3

from depthai_sdk import Camera

with Camera() as cam:
    rgb_camera = cam.create_camera(out=True)
    face_det = cam.create_nn("face-detection-retail-0004", rgb_camera, spatials=True, out=True)
    # If input from an object detection NN, crop around BB from HQ frames for new model
    age_gender_nn = cam.create_nn("age_gender_recognition_0001", input=face_det, out=True)

    cam.start()
    
    face_visualizer = cam.create_visualizer(rgb_camera, face_det, age_gender_nn)

    while cam.running():
        msgs = cam.get_synced_msgs()
        face_visualizer.visualize(msgs)

        cam.poll()
