#!/usr/bin/env python3

import cv2
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode

'''
FastDepth demo running on device.
https://github.com/zl548/MegaDepth


Run as:
python3 -m pip install -r requirements.txt
python3 people-detection.py

Onnx taken from PINTO0309, added scaling flag, and exported to blob:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/153_MegaDepth
'''

NN_WIDTH, NN_HEIGHT = 256, 192


def callback(packet):
    frame = packet.frame
    nn_data = packet.img_detections

    pred = np.array(nn_data.getFirstLayerFp16()).reshape((NN_HEIGHT, NN_WIDTH))

    # Scale depth to get relative depth
    d_min = np.min(pred)
    d_max = np.max(pred)
    depth_relative = (pred - d_min) / (d_max - d_min)

    # Color it
    depth_relative = np.array(depth_relative) * 255
    depth_relative = depth_relative.astype(np.uint8)
    depth_relative = 255 - depth_relative
    depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)

    target_width = int(1920)
    target_height = int(NN_HEIGHT * (1920 / NN_HEIGHT) / (16 / 9))

    frame = cv2.resize(packet.frame, (target_width, target_height))
    depth_relative = cv2.resize(depth_relative, (target_width, target_height))

    cv2.imshow('Mega-depth', cv2.hconcat([frame, depth_relative]))


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    nn_path = 'models/megadepth_192x256_openvino_2021.4_6shave.blob'
    nn = oak.create_nn(nn_path, color)
    nn.config_nn(aspect_ratio_resize_mode=AspectRatioResizeMode.STRETCH)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
