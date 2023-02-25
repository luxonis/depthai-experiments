import cv2
import numpy as np

from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket


def get_frame(img, shape):
    return np.array(img.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8)


def callback(packet: DetectionPacket):
    edge_frame = get_frame(packet.img_detections, (3, 300, 300))

    cv2.imshow('Laplacian edge detection', edge_frame)
    cv2.imshow('Color', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    color.config_color_camera(interleaved=False)
    nn = oak.create_nn('models/edge_simplified_openvino_2021.4_6shave.blob', color)
    nn.config_nn(resize_mode='crop')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
