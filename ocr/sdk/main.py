import blobconverter
import cv2
import numpy as np

import east
from codec import codec
from depthai_sdk import OakCamera, ResizeMode
from depthai_sdk.classes import Detections
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox

NN_WIDTH = 256
NN_HEIGHT = 256


def decode(nn_data):
    scores, geom1, geom2 = [nn_data.getLayerFp16(name) for name in nn_data.getAllLayerNames()]

    scores = np.reshape(scores, (1, 1, 64, 64))
    geom1 = np.reshape(geom1, (1, 4, 64, 64))
    geom2 = np.reshape(geom2, (1, 1, 64, 64))

    bboxes, confs, angles = east.decode_predictions(scores, geom1, geom2)
    boxes, angles = east.non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))

    rotated_rectangles = [
        east.get_cv_rotated_rect(bbox, angle * -1)
        for (bbox, angle) in zip(boxes, angles)
    ]

    dets = Detections(nn_data=nn_data, is_rotated=True)
    bboxes = []
    for rect in rotated_rectangles:
        rect_w, rect_h = rect[1]
        x_min = (rect[0][0] - rect_w / 2) / NN_WIDTH
        y_min = (rect[0][1] - rect_h / 2) / NN_HEIGHT
        x_max = (rect[0][0] + rect_w / 2) / NN_WIDTH
        y_max = (rect[0][1] + rect_h / 2) / NN_HEIGHT
        bbox = (x_min, y_min, x_max, y_max)
        angle = rect[2]
        bboxes.append(bbox)
        dets.add(0, 1.0, bbox, angle=angle)

    return dets


def decode_recognition(nn_data):
    rec_data = np.array(nn_data.getFirstLayerFp16()).reshape(30, 1, 37)
    decoded_text = codec.decode(rec_data)
    return decoded_text


def callback(packet):
    frame = packet.frame

    detections = packet.img_detections.detections

    texts = []
    for i, det in enumerate(detections):
        bbox = det.xmin, det.ymin, det.xmax, det.ymax
        bbox = NormalizeBoundingBox((NN_WIDTH, NN_HEIGHT), ResizeMode.STRETCH).normalize(packet.frame, bbox)
        bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        rect = (
            [float(bbox[0]), float(bbox[1])],
            [float(bbox_w), float(bbox_h)],
            packet.img_detections.angles[i]
        )

        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        texts.append(cv2.resize(packet.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (240, 64)))
        cv2.polylines(frame, [rect], True, (0, 255, 0), 2)

    try:
        print(packet.nnData)
    except AttributeError:
        pass

    if len(texts) > 0:
        cv2.imshow('Texts', np.vstack(texts))

    cv2.imshow('Detections', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', fps=10)
    version = '2021.4'

    # Download the models
    det_blob_path = blobconverter.from_zoo(name='east_text_detection_256x256',
                                           zoo_type='depthai',
                                           shaves=7,
                                           version=version)
    rec_blob_path = blobconverter.from_zoo(name='text-recognition-0012',
                                           shaves=7,
                                           version=version)

    # Create NN components
    det_nn = oak.create_nn(det_blob_path, color, decode_fn=decode)
    det_nn.config_nn(resize_mode='stretch')

    rec_nn = oak.create_nn(rec_blob_path, input=det_nn, decode_fn=decode_recognition)

    oak.callback(rec_nn, callback=callback)

    # oak.callback(rec_nn, callback=callback)
    oak.start(blocking=True)
