import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket

THRESHOLD = 0.2
NN_WIDTH = 192
NN_HEIGHT = 192
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

colors_full = np.random.randint(255, size=(100, 3), dtype=int)


def plot_boxes(frame, boxes, colors, scores):
    color_black = (0, 0, 0)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        y1 = (frame.shape[0] * box[0]).astype(int)
        y2 = (frame.shape[0] * box[2]).astype(int)
        x1 = (frame.shape[1] * box[1]).astype(int)
        x2 = (frame.shape[1] * box[3]).astype(int)
        color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 15), color, -1)
        cv2.putText(frame, f"{scores[i]:.2f}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color_black)


def callback(packet: DetectionPacket):
    frame = packet.frame
    nn_data = packet.img_detections

    # get outputs
    detection_boxes = np.array(nn_data.getLayerFp16("ExpandDims")).reshape((100, 4))
    detection_scores = np.array(nn_data.getLayerFp16("ExpandDims_2")).reshape((100,))

    # keep boxes bigger than threshold
    mask = detection_scores >= THRESHOLD
    boxes = detection_boxes[mask]
    colors = colors_full[mask]
    scores = detection_scores[mask]

    # draw boxes
    plot_boxes(frame, boxes, colors, scores)

    cv2.imshow("Localizer", frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')

    nn_path = blobconverter.from_zoo(name="mobile_object_localizer_192x192", zoo_type="depthai", shaves=6)
    nn = oak.create_nn(nn_path, color)
    nn.config_nn(resize_mode='stretch')

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
