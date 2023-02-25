import cv2
import numpy as np

from depthai_sdk import OakCamera, ResizeMode
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox
from utils.utils import get_boxes

MAX_CANDIDATES = 75
MIN_SIZE = 1
THRESH = 0.01
BOX_THRESH = 0.2

NN_WIDTH, NN_HEIGHT = 640, 480


def callback(packet):
    nn_data = packet.img_detections
    pred = np.array(nn_data.getLayerFp16("out")).reshape((NN_HEIGHT, NN_WIDTH))
    boxes, scores = get_boxes(pred, THRESH, BOX_THRESH, MIN_SIZE, MAX_CANDIDATES)

    for bbox in boxes:
        bbox = (bbox[0, 0], bbox[0, 1], bbox[2, 0], bbox[2, 1])
        bbox = NormalizeBoundingBox((NN_WIDTH, NN_HEIGHT), ResizeMode.LETTERBOX).normalize(packet.frame, bbox)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        x1, x2 = np.clip([x1, x2], 0, packet.frame.shape[1])
        y1, y2 = np.clip([y1, y2], 0, packet.frame.shape[0])

        packet.frame[y1:y2, x1:x2] = cv2.GaussianBlur(packet.frame[y1:y2, x1:x2], (49, 49), 30)

    cv2.imshow("Text blurring", packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', fps=30)
    det_nn = oak.create_nn('models/text_detection_db_480x640_openvino_2021.4_6shave.blob', color)

    oak.callback(det_nn, callback=callback)

    oak.start(blocking=True)
