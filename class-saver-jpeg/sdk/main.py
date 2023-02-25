import csv
import time
from pathlib import Path

import cv2
from depthai_sdk import OakCamera

labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Create data folder
Path('data').mkdir(parents=True, exist_ok=True)


def create_folders():
    Path(f'data/raw').mkdir(parents=True, exist_ok=True)
    for text in labels:
        Path(f'data/{text}').mkdir(parents=True, exist_ok=True)


def callback(packet, visualizer):
    original_frame = packet.frame

    timestamp = int(time.time() * 10000)
    raw_frame_path = f'data/raw/{timestamp}.jpg'
    cv2.imwrite(raw_frame_path, original_frame)

    frame = visualizer.draw(original_frame.copy())

    for detection in packet.detections:
        bbox = (*detection.top_left, *detection.bottom_right)

        det_frame = original_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        cropped_path = f'data/{detection.label}/{timestamp}_cropped.jpg'
        overlay_path = f'data/{detection.label}/{timestamp}_overlay.jpg'

        cv2.imwrite(cropped_path, det_frame)
        cv2.imwrite(overlay_path, frame)

        data = {
            'timestamp': timestamp,
            'label': detection.label,
            'left': bbox[0],
            'top': bbox[1],
            'right': bbox[2],
            'bottom': bbox[3],
            'raw_frame': raw_frame_path,
            'overlay_frame': overlay_path,
            'cropped_frame': cropped_path,
        }
        dataset.writerow(data)

    cv2.imshow('Class saver', frame)


with OakCamera() as oak, open('data/dataset.csv', 'w') as csvfile:
    create_folders()

    dataset = csv.DictWriter(csvfile, ['timestamp', 'label', 'left', 'top', 'right', 'bottom',
                                       'raw_frame', 'overlay_frame', 'cropped_frame'])
    dataset.writeheader()

    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)

    oak.visualize(nn.out.main, callback=callback, fps=True)
    oak.start(blocking=True)
