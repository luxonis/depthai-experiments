import argparse

import cv2
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode, DetectionPacket
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox
from utils.utils import get_boxes

parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn_model", help="select model path for inference",
                    default='models/text_detection_db_480x640_openvino_2021.4_6shave.blob', type=str)
parser.add_argument("-bt", "--box_thresh", help="set the confidence threshold of boxes", default=0.2, type=float)
parser.add_argument("-t", "--thresh", help="set the bitmap threshold", default=0.01, type=float)
parser.add_argument("-ms", "--min_size", default=1, type=int, help='set min size of box')
parser.add_argument("-mc", "--max_candidates", default=75, type=int, help='maximum number of candidate boxes')

args = parser.parse_args()

NN_PATH = args.nn_model
MAX_CANDIDATES = args.max_candidates
MIN_SIZE = args.min_size
THRESH = args.thresh
BOX_THRESH = args.box_thresh

NN_WIDTH, NN_HEIGHT = 640, 480


def callback(packet: DetectionPacket):
    nn_data = packet.img_detections
    pred = np.array(nn_data.getLayerFp16("out")).reshape((NN_HEIGHT, NN_WIDTH))
    boxes, scores = get_boxes(pred, THRESH, BOX_THRESH, MIN_SIZE, MAX_CANDIDATES)

    blur = cv2.GaussianBlur(packet.frame, (49, 49), 30)

    for bbox in boxes:
        bbox = (bbox[0, 0], bbox[0, 1], bbox[2, 0], bbox[2, 1])
        bbox = NormalizeBoundingBox((NN_WIDTH, NN_HEIGHT), AspectRatioResizeMode.LETTERBOX).normalize(packet.frame, bbox)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        x1, x2 = np.clip([x1, x2], 0, packet.frame.shape[1])
        y1, y2 = np.clip([y1, y2], 0, packet.frame.shape[0])

        packet.frame[y1:y2, x1:x2] = blur[y1:y2, x1:x2]

    cv2.imshow("Text blurring", packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', fps=30)
    det_nn = oak.create_nn(NN_PATH, color)

    oak.callback(det_nn, callback=callback)

    oak.start(blocking=True)
