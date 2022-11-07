import blobconverter
import cv2
import numpy as np

from depthai_sdk import OakCamera, DetectionPacket, AspectRatioResizeMode, Visualizer, TextPosition
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox


# TODO - change to better detector
detector = cv2.QRCodeDetector()


def expand_detection(det, percent=2):
    percent /= 100
    det.xmin = np.clip(det.xmin - percent, 0, 1)
    det.ymin = np.clip(det.ymin - percent, 0, 1)
    det.xmax = np.clip(det.xmax + percent, 0, 1)
    det.ymax = np.clip(det.ymax + percent, 0, 1)


def callback(packet: DetectionPacket, visualizer: Visualizer, **kwargs):
    for i, detection in enumerate(packet.img_detections.detections):
        expand_detection(detection)
        bbox = detection.xmin, detection.ymin, detection.xmax, detection.ymax
        bbox = NormalizeBoundingBox((384, 384), AspectRatioResizeMode.LETTERBOX).normalize(packet.frame, bbox)

        cropped_qr = packet.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        data, _, _ = detector.detectAndDecode(cropped_qr)

        if data:
            visualizer.add_text(data, color=(232, 36, 87),
                                bbox=bbox, position=TextPosition.TOP_MID)

    cv2.imshow('QR example', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', '1080p')
    nn_path = blobconverter.from_zoo(name="qr_code_detection_384x384", zoo_type="depthai", shaves=6)
    nn = oak.create_nn(nn_path, color, type='mobilenet')
    nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.LETTERBOX)

    # oak.callback(nn, callback=callback)
    visualizer = oak.visualize(nn, callback=callback, fps=True)
    visualizer.text(auto_scale=True).detections(hide_label=True)

    oak.start(blocking=True)
