import cv2

from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket


def callback(packet: DetectionPacket):
    cv2.imshow('NN output', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    raise NotImplementedError('TODO')

    nn = oak.create_nn('models/diff_openvino_2021.4_6shave.blob', color)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
