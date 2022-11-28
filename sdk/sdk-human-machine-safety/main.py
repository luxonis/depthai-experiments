import blobconverter
import cv2

from depthai_sdk import OakCamera
from palm_detection import PalmDetection

pd = PalmDetection()


def callback(sync_dict: dict):
    palm_detection_packet = sync_dict['2_out;0_preview']
    mobilenet_packet = sync_dict['3_out;0_preview']

    frame = pd.run_palm(palm_detection_packet.frame, palm_detection_packet.img_detections)

    cv2.imshow('Machine safety', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')
    stereo = oak.create_stereo(resolution='800p')

    raise NotImplementedError('TODO')

    palm_nn_path = blobconverter.from_zoo(name='palm_detection_128x128', shaves=6, zoo_type='depthai')
    palm_detection = oak.create_nn(palm_nn_path, color, spatial=stereo, nn_type='mobilenet')

    mobilenet = oak.create_nn('mobilenet-ssd', color, spatial=stereo)

    oak.sync([palm_detection.out.main, mobilenet.out.main], callback=callback)
    oak.start(blocking=True)
