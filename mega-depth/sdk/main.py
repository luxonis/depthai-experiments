import cv2
import numpy as np

from depthai_sdk import OakCamera

NN_WIDTH, NN_HEIGHT = 256, 192


def callback(packet):
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
    nn.config_nn(resize_mode='stretch')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
