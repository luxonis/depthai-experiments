import cv2
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox

CONFIDENCE = 0.7


def callback(packet, visualizer):
    nn_data = packet.img_detections
    bboxes = np.array(nn_data.getFirstLayerFp16())
    bboxes = bboxes.reshape((bboxes.size // 7, 7))

    bboxes = bboxes[bboxes[:, 2] > CONFIDENCE][:, 3:7]

    for bbox in bboxes:
        bbox = NormalizeBoundingBox((384, 384), AspectRatioResizeMode.LETTERBOX).normalize(packet.frame, bbox)
        cv2.rectangle(packet.frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (232, 36, 87), 2)

    cv2.imshow('Gaze estimation', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    raise NotImplementedError('TODO: Multiple inputs are not supported yet')

    face_detection = oak.create_nn('face-detection-retail-0004', color)
    gaze_estimation = oak.create_nn('gaze-estimation-adas-0002', face_detection)

    oak.callback(gaze_estimation, callback=callback)
    oak.start(blocking=True)
