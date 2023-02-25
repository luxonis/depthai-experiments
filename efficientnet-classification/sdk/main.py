import cv2
import numpy as np

from classes import class_names
from depthai_sdk import OakCamera

labels = class_names()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def callback(packet):
    nn_data = packet.img_detections

    data = softmax(nn_data.getFirstLayerFp16())
    result_conf = np.max(data)

    result = {
        "name": labels[np.argmax(data)],
        "conf": round(100 * result_conf, 2)
    }
    cv2.putText(packet.frame, f"{result['name']} {result['conf']:.2f}%",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('EfficientNet-b0 classification', packet.frame)


with OakCamera(replay='animals.mp4') as oak:
    color = oak.create_camera('color')

    nn = oak.create_nn('efficientnet-b0', color)

    oak.callback(nn.out.main, callback)
    oak.start(blocking=True)
