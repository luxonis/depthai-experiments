import cv2
import numpy as np

from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket

MAX_DISTANCE = 1000


def calculate_mutual_distance(points):
    """
    Calculate the distance between all pairs of points.
    The result is a symmetric matrix of shape (n, n) where n is the number of points.
    """
    coords = np.array([[p.x, p.y, p.z] for p in points])
    return np.sqrt(np.power(coords[:, :, None] - coords[:, :, None].T, 2).sum(axis=1))


def callback(packet: DetectionPacket):
    visualizer = packet.visualizer
    bboxes = []
    spatial_coords = []

    for detection, spatial_detection in zip(packet.detections, packet.img_detections.detections):
        bbox = (*detection.top_left, *detection.bottom_right)
        bboxes.append(bbox)
        spatial_coords.append(spatial_detection.spatialCoordinates)

    # If there are at least two people detected, calculate the distance between them
    if len(spatial_coords) > 1:
        mutual_distance = calculate_mutual_distance(spatial_coords)
        lower_tril = np.tril(mutual_distance, -1)
        below_threshold = (lower_tril > 0) & (lower_tril < MAX_DISTANCE)

        indices = np.nonzero(below_threshold)
        dangerous_pairs = list(zip(indices[0], indices[1]))
        unique_pairs = set(tuple(sorted(p)) for p in dangerous_pairs)
        overlay = packet.frame.copy()

        # Draw a line between people who are too close
        for pair in unique_pairs:
            bbox1, bbox2 = bboxes[pair[0]], bboxes[pair[1]]
            x1, y1 = (bbox1[0] + bbox1[2]) // 2, int(bbox1[3])
            x2, y2 = (bbox2[0] + bbox2[2]) // 2, int(bbox2[3])

            danger_color = (0, 0, 255)

            visualizer.add_line((x1, y1), (x2, y2), danger_color, 2)

            w1 = bbox1[2] - bbox1[0]
            w2 = bbox2[2] - bbox2[0]
            cv2.ellipse(overlay, (x1, y1), (w1 // 2, 10), 0, 0, 360, danger_color, thickness=cv2.FILLED)
            cv2.ellipse(overlay, (x2, y2), (w2 // 2, 10), 0, 0, 360, danger_color, thickness=cv2.FILLED)

        cv2.addWeighted(overlay, 0.5, packet.frame, 0.5, 0, packet.frame)

    visualizer.draw(packet.frame)
    cv2.imshow('Social distancing', packet.frame)


with OakCamera() as oak:
    color = oak.create_camera('color', '1080p', fps=30)

    stereo = oak.create_stereo('800p', fps=30)
    stereo.config_stereo(confidence=255)

    det_nn = oak.create_nn('person-detection-retail-0013', color, spatial=stereo)

    oak.visualize([det_nn], callback=callback, fps=True)

    oak.start(blocking=True)
