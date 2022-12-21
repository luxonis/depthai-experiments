import blobconverter
import cv2

import face_landmarks
from depthai_sdk import OakCamera, TwoStagePacket
from depthai_sdk.callback_context import CallbackContext

decode = face_landmarks.FaceLandmarks()


def callback(ctx: CallbackContext):
    packet: TwoStagePacket = ctx.packet

    for nn_data, detection in zip(packet.nnData, packet.img_detections.detections):
        bbox = detection.xmin, detection.ymin, detection.xmax, detection.ymax
        decode.run_land68(packet.frame, nn_data, bbox)

    cv2.imshow('Fatigue Detection', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    face_detection = oak.create_nn('face-detection-retail-0004', color)

    landmark_nn_path = blobconverter.from_zoo(name='facial_landmarks_68_160x160',
                                              shaves=6,
                                              zoo_type='depthai',
                                              version='2021.4')
    landmark_detection = oak.create_nn(landmark_nn_path, face_detection)

    oak.callback(landmark_detection, callback=callback)
    oak.start(blocking=True)
