import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera

NN_WIDTH, NN_HEIGHT = 320, 240


def callback(packet):
    nn_data = packet.img_detections

    # Get output layer
    pred = np.array(nn_data.getFirstLayerFp16()).reshape((NN_HEIGHT // 2, NN_WIDTH // 2))

    # Scale depth to get relative depth
    d_min = np.min(pred)
    d_max = np.max(pred)
    depth_relative = (pred - d_min) / (d_max - d_min)

    # Color it
    depth_relative = np.array(depth_relative) * 255
    depth_relative = depth_relative.astype(np.uint8)
    depth_relative = cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)

    # Resize to match color frame
    depth_relative = cv2.resize(depth_relative, (packet.frame.shape[1], packet.frame.shape[0]))

    # Concatenate NN input and produced depth
    cv2.imshow('Depth MobileNetV2', cv2.hconcat([packet.frame, depth_relative]))


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    nn_path = blobconverter.from_zoo(name="depth_estimation_mbnv2_240x320", zoo_type="depthai", shaves=6)
    nn = oak.create_nn(nn_path, color)
    nn.config_nn(resize_mode='stretch')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
