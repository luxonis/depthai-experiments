import cv2
import numpy as np

from depthai_sdk import OakCamera, ResizeMode
from depthai_sdk.classes import TwoStagePacket
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox
from functions import draw_box

OBJECTRON_CHAIR_PATH = 'models/objectron_chair_openvino_2021.4_6shave.blob'

INPUT_W, INPUT_H = 640, 360


def callback(packet: TwoStagePacket):
    frame = packet.frame

    detections = packet.img_detections.detections
    # find the detection with the highest confidence
    # label of 9 indicates chair in VOC
    confs = [det.confidence for det in detections if det.label == 9]

    if len(confs) > 0:
        idx = confs.index(max(confs))
        detection = detections[idx]

        out = np.array(packet.nnData[idx].getLayerFp16("StatefulPartitionedCall:1")).reshape(9, 2)

        bbox = detection.xmin, detection.ymin, detection.xmax, detection.ymax
        x_min, y_min, x_max, y_max = NormalizeBoundingBox((224, 224), ResizeMode.CROP).normalize(frame, bbox)

        OG_W, OG_H = x_max - x_min, y_max - y_min

        scale_x = OG_W / 224
        scale_y = OG_H / 224

        out[:, 0] = out[:, 0] * scale_x + x_min
        out[:, 1] = out[:, 1] * scale_y + y_min

        draw_box(frame, out)

    frame = cv2.resize(frame, (INPUT_W, INPUT_H))
    cv2.imshow('Objectron', frame)


with OakCamera() as oak:
    color = oak.create_camera('color')

    mobilenet = oak.create_nn('mobilenet-ssd', color)
    mobilenet.config_nn(resize_mode='crop')

    objectron = oak.create_nn(OBJECTRON_CHAIR_PATH, mobilenet)

    oak.callback(objectron.out.main, callback=callback)
    oak.start(blocking=True)
