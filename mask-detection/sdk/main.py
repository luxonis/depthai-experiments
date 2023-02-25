import cv2
import numpy as np

from depthai_sdk import OakCamera
from depthai_sdk.classes import TwoStagePacket
from depthai_sdk.visualize.configs import TextPosition


def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def callback(packet: TwoStagePacket):
    visualizer = packet.visualizer
    for det, rec in zip(packet.detections, packet.nnData):
        index = np.argmax(log_softmax(rec.getFirstLayerFp16()))
        text = "No Mask"
        color = (0, 0, 255)  # Red
        if index == 1:
            text = "Mask"
            color = (0, 255, 0)

        visualizer.add_text(text,
                            bbox=(*det.top_left, *det.bottom_right),
                            position=TextPosition.BOTTOM_MID,
                            color=color)

    frame = visualizer.draw(packet.frame)
    cv2.imshow('Mask detection', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    det = oak.create_nn('face-detection-retail-0004', color)
    det.config_nn(resize_mode='crop')

    mask = oak.create_nn('sbd_mask_classification_224x224', input=det)

    # Visualize detections on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(mask, callback=callback).detections(fill_transparency=0.1)
    oak.visualize(det.out.passthrough)

    oak.start(blocking=True)  # This call will block until the app is stopped (by pressing 'Q' button)
