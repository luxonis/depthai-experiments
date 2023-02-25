import cv2
import numpy as np

from depthai_sdk import OakCamera

CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def callback(packet):
    nn_data = packet.img_detections
    logits = softmax(nn_data.getFirstLayerFp16())
    pred = np.argmax(logits)
    conf = logits[pred]

    if conf > 0.5:
        cv2.putText(packet.frame, f'{CLASSES[pred]}: {conf:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Flower classification', packet.frame)


with OakCamera(replay='input.mp4') as oak:
    color = oak.create_camera('color')
    flower_nn = oak.create_nn('flower.blob', color)

    oak.callback(flower_nn, callback=callback)
    oak.start(blocking=True)
