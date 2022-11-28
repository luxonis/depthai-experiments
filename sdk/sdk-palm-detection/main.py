import blobconverter
import cv2

from depthai_sdk import OakCamera
from depthai_sdk.callback_context import CallbackContext
from palm_detection import PalmDetection

palm_detection = PalmDetection()


def callback(ctx: CallbackContext):
    packet = ctx.packet

    frame = packet.frame
    nn_data = packet.img_detections

    palm_coords = palm_detection.decode(frame, nn_data)
    for bbox in palm_coords:
        cv2.rectangle(
            img=frame,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=(0, 127, 255),
            thickness=4
        )

    cv2.imshow('Palm detection', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    nn_path = blobconverter.from_zoo(name='palm_detection_128x128', zoo_type='depthai', shaves=6)
    model_nn = oak.create_nn(nn_path, color)

    oak.callback(model_nn, callback=callback)
    oak.start(blocking=True)
