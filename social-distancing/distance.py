import itertools
import logging
import math
import cv2

log = logging.getLogger(__name__)


def calculate_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distance


class DistanceGuardian:
    max_distance = 1

    def parse_frame(self, frame, boxes):
        results = []
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                point1 = box1['distance_x'], box1['distance_y'], box1['distance_z']
                point2 = box2['distance_x'], box2['distance_y'], box2['distance_z']
                distance = calculate_distance(point1, point2)
                log.info("DG: {}".format(distance))
                results.append({
                    'distance': distance,
                    'dangerous': distance < self.max_distance,
                    'box1': box1,
                    'box2': box2,
                })

        return results


class DistanceGuardianDebug(DistanceGuardian):
    def parse_frame(self, frame, boxes):
        results = super().parse_frame(frame, boxes)
        overlay = frame.copy()
        for result in results:
            x1 = int(result['box1']['left'] + (result['box1']['right'] - result['box1']['left']) / 2)
            y1 = int(result['box1']['bottom'])
            x2 = int(result['box2']['left'] + (result['box2']['right'] - result['box2']['left']) / 2)
            y2 = int(result['box2']['bottom'])
            color = (0, 0, 255) if result['dangerous'] else (255, 0, 0)
            cv2.ellipse(overlay, (x1, y1), (40, 10), 0, 0, 360, color, thickness=cv2.FILLED)
            cv2.ellipse(overlay, (x2, y2), (40, 10), 0, 0, 360, color, thickness=cv2.FILLED)
            cv2.line(overlay, (x1, y1), (x2, y2), color, 1)
            label_size, baseline = cv2.getTextSize(str(round(result['distance'], 1)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 1)
            label_x = (x1 + x2 - label_size[0]) // 2
            label_y = (y1 + y2 - label_size[1]) // 2
            cv2.putText(overlay, str(round(result['distance'], 1)), (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)

        frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        return results
