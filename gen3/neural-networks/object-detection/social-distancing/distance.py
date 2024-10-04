import math

import cv2

MAX_DISTANCE = 1

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.5


def calculate_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance


def parse_distance(frame, detections):
    overlay = frame.copy()
    results = []

    for i, detection1 in enumerate(detections):
        for detection2 in detections[i + 1 :]:
            point1 = detection1["depth_x"], detection1["depth_y"], detection1["depth_z"]
            point2 = detection2["depth_x"], detection2["depth_y"], detection2["depth_z"]
            distance = calculate_distance(point1, point2)
            dangerous = distance < MAX_DISTANCE

            results.append(
                {
                    "distance": distance,
                    "dangerous": dangerous,
                    "detection1": detection1,
                    "detection2": detection2,
                }
            )

            x1 = detection1["x_min"] + (detection1["x_max"] - detection1["x_min"]) // 2
            y1 = detection1["y_max"]
            x2 = detection2["x_min"] + (detection2["x_max"] - detection2["x_min"]) // 2
            y2 = detection2["y_max"]
            color = (0, 0, 255) if dangerous else (255, 0, 0)

            cv2.ellipse(
                overlay, (x1, y1), (40, 10), 0, 0, 360, color, thickness=cv2.FILLED
            )
            cv2.ellipse(
                overlay, (x2, y2), (40, 10), 0, 0, 360, color, thickness=cv2.FILLED
            )
            cv2.line(overlay, (x1, y1), (x2, y2), color, 1)
            label_size, baseline = cv2.getTextSize(
                str(round(distance, 1)), FONT, 0.5, 1
            )
            label_x = (x1 + x2 - label_size[0]) // 2
            label_y = (y1 + y2 - label_size[1]) // 2
            cv2.putText(
                overlay,
                str(round(distance, 1)),
                (label_x, label_y),
                FONT,
                0.5,
                color,
                1,
            )

    frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    return results
