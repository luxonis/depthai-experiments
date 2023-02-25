import cv2
import numpy as np

from depthai_sdk import OakCamera


def get_frame(img_frame, shape):
    return np.array(img_frame.getData()).view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8)


def callback(packet):
    edge_frame = get_frame(packet.img_detections, (3, 300, 300))

    cv2.imshow('Blurred image', edge_frame)
    cv2.imshow('Color', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    nn = oak.create_nn('models/blur_simplified_openvino_2021.4_6shave.blob', color)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
