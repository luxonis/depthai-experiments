import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera, AspectRatioResizeMode, TextPosition
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox

# TODO - change to better detector
detector = cv2.QRCodeDetector()


def expand_detection(det, percent=2):
    percent /= 100
    det.xmin = np.clip(det.xmin - percent, 0, 1)
    det.ymin = np.clip(det.ymin - percent, 0, 1)
    det.xmax = np.clip(det.xmax + percent, 0, 1)
    det.ymax = np.clip(det.ymax + percent, 0, 1)


def callback(packet, visualizer):
    for i, detection in enumerate(packet.img_detections.detections):
        expand_detection(detection)
        bbox = detection.xmin, detection.ymin, detection.xmax, detection.ymax
        bbox = NormalizeBoundingBox((384, 384), AspectRatioResizeMode.LETTERBOX).normalize(np.zeros(visualizer.frame_shape), bbox)
        cropped_qr = packet.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        data, _, _ = detector.detectAndDecode(cropped_qr)

        print(data)

    frame = visualizer.draw(packet.frame)
    cv2.imshow('QR scanner', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', '1080p', encode='h264')

    nn_path = blobconverter.from_zoo(name="qr_code_detection_384x384", zoo_type="depthai", shaves=6)
    nn = oak.create_nn(nn_path, color, nn_type='mobilenet')

    # visualizer = oak.visualize(nn, callback=callback, fps=True)
    # visualizer.text(auto_scale=True).detections(hide_label=True)
    oak.callback(nn, callback=callback)

    oak.start(blocking=True)
