import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera, DetectionPacket, AspectRatioResizeMode
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox

CONFIDENCE = 0.7


def callback(packet: DetectionPacket):
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

    face_detection_path = blobconverter.from_zoo(name='face-detection-retail-0004', shaves=4)

    face_detection = oak.create_nn(face_detection_path, color)
    face_detection.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.LETTERBOX)

    # TODO problem with multistage - doesn't support non mobilenet/yolo nets
    gaze_estimation_path = blobconverter.from_zoo(name='gaze-estimation-adas-0002', shaves=4)
    gaze_estimation = oak.create_nn(gaze_estimation_path, face_detection)

    oak.callback(gaze_estimation, callback=callback)
    oak.start(blocking=True)
