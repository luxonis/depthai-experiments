import blobconverter
import cv2
from pyzbar import pyzbar

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import TextPosition


def callback(packet):
    for detection in packet.detections:
        bbox = detection.top_left[0], detection.top_left[1], detection.bottom_right[0], detection.bottom_right[1]
        bbox = [max(0, bbox[0] - 10), max(0, bbox[1] - 10), min(packet.frame.shape[1], bbox[2] + 10), min(packet.frame.shape[0], bbox[3] + 10)]
        cropped_qr = packet.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        barcodes = pyzbar.decode(cropped_qr)
        for barcode in barcodes:
            barcode_info = barcode.data.decode('utf-8')
            packet.visualizer.add_text(barcode_info, bbox=bbox, position=TextPosition.MID, outline=True)

    frame = packet.visualizer.draw(packet.frame)
    cv2.imshow('QR code recognition', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', '1080p', fps=10)

    nn_path = blobconverter.from_zoo(name="qr_code_detection_384x384", zoo_type="depthai")
    nn = oak.create_nn(nn_path, color, nn_type='mobilenet')

    visualizer = oak.visualize(nn.out.main, callback=callback)
    visualizer.detections(hide_label=True, thickness=3).text(auto_scale=False)
    oak.start(blocking=True)
