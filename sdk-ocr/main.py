import blobconverter
import cv2
import numpy as np
from depthai import NNData

import east
from depthai_sdk import OakCamera, DetectionPacket


def decode_det(data: NNData):
    return data.getFirstLayerFp16()


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def callback(packet: DetectionPacket):
    nn_data = packet.img_detections

    if not nn_data:
        return

    scores, geom1, geom2 = to_tensor_result(nn_data).values()
    scores = np.reshape(scores, (1, 1, 64, 64))
    geom1 = np.reshape(geom1, (1, 4, 64, 64))
    geom2 = np.reshape(geom2, (1, 1, 64, 64))

    bboxes, confs, angles = east.decode_predictions(scores, geom1, geom2)
    boxes, angles = east.non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
    rotated_rectangles = [
        east.get_cv_rotated_rect(bbox, angle * -1)
        for (bbox, angle) in zip(boxes, angles)
    ]

    h, w = packet.frame.shape[:2]
    scale_h, scale_w = h / 256, w / 256
    for rotated_rect in rotated_rectangles:
        rotated_rect[0][0] = rotated_rect[0][0] * scale_w
        rotated_rect[0][1] = rotated_rect[0][1] * scale_h
        rotated_rect[1][0] = rotated_rect[1][0] * scale_w
        rotated_rect[1][1] = rotated_rect[1][1] * scale_h

        box = np.int0(cv2.boxPoints(rotated_rect))
        cv2.polylines(packet.frame, [box], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

    cv2.imshow('OCR', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', fps=30)
    version = '2021.4'

    # Download the models
    det_blob_path = blobconverter.from_zoo(name='east_text_detection_256x256',
                                           zoo_type='depthai',
                                           shaves=6,
                                           version=version)
    rec_blob_path = blobconverter.from_zoo(name='text-recognition-0012',
                                           shaves=6,
                                           version=version)

    # Create NN components
    det_nn = oak.create_nn(det_blob_path, color)
    # det_nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.CROP)

    # rec_nn = oak.create_nn(rec_blob_path, input=det_nn)
    oak.callback(det_nn.out.main, callback=callback)
    # oak.callback(rec_nn, callback=callback)
    oak.start(blocking=True)
