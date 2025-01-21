#!/usr/bin/env python3
from depthai_sdk import OakCamera

# Here, instead of using one of the public depthai recordings
# https://docs.luxonis.com/projects/sdk/en/latest/features/replaying/#public-depthai-recordings
# We can specify path to our recording, eg. OakCamera(replay='recordings/1-184430102127631200')
with OakCamera(replay='people-tracking-above-05') as oak:
    oak.replay.set_loop(True)
    left = oak.create_camera('CAM_A') # CAM_A.mp4
    right = oak.create_camera('CAM_C') # CAM_C.mp4

    # TODO: Use a better suited model that was specifically trained on top-down view images of people.
    nn = oak.create_nn('yolov8n_coco_640x352', right, tracker=True)

    stereo = oak.create_stereo(left=left, right=right)
    stereo.config_stereo(lr_check=True)
    oak.visualize([stereo.out.rectified_right], fps=True)
    oak.visualize(stereo.out.depth, fps=True)

    oak.visualize(nn, fps=True)
    oak.start(blocking=True)
