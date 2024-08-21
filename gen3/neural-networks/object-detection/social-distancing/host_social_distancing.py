import cv2
import depthai as dai
import numpy as np
from distance import parse_distance
from alerting import AlertingGate
import math

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0, 0, 255)

MAX_Z = 4
MIN_Z = 1
MAX_X = 0.9
MIN_X = -0.7

BIRD_FRAME_X = 100
BIRD_FRAME_Z = 320

class SocialDistancing(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.alerting = AlertingGate()
        self.distance_bird_frame = self.make_bird_frame()


    def build(self, preview: dai.Node.Output, nn: dai.Node.Output) -> "SocialDistancing":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, preview: dai.ImgFrame, detections: dai.SpatialImgDetections) -> None:
        frame = preview.getCvFrame()
        bird_frame = self.distance_bird_frame.copy()
        height = frame.shape[0]
        width = frame.shape[1]

        bboxes = []
        for detection in detections.detections:
            label = detection.label
            confidence = detection.confidence
            x_min = int(detection.xmin * width)
            x_max = int(detection.xmax * width)
            y_min = int(detection.ymin * height)
            y_max = int(detection.ymax * height)
            depth_x = detection.spatialCoordinates.x / 1000
            depth_y = detection.spatialCoordinates.y / 1000
            depth_z = detection.spatialCoordinates.z / 1000

            bboxes.append({
                'label': label,
                'confidence': confidence,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'depth_x': depth_x,
                'depth_y': depth_y,
                'depth_z': depth_z
            })

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "x: {}".format(round(depth_x, 1)), (x_min, y_min + 30), FONT, 0.5, COLOR)
            cv2.putText(frame, "y: {}".format(round(depth_y, 1)), (x_min, y_min + 50), FONT, 0.5, COLOR)
            cv2.putText(frame, "z: {}".format(round(depth_z, 1)), (x_min, y_min + 70), FONT, 0.5, COLOR)
            cv2.putText(frame, "conf: {}".format(round(confidence, 1)), (x_min, y_min + 90), FONT, 0.5, COLOR)
            cv2.putText(frame, "label: {}".format(label, 1), (x_min, y_min + 110), FONT, 0.5, COLOR)

            left, right = self.calc_x(depth_x)
            top, bottom = self.calc_z(depth_z)
            cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        distance_results = parse_distance(frame, bboxes)
        should_alert = self.alerting.parse_danger(distance_results)
        
        if should_alert:
            cv2.putText(frame, "Too close", (int(width / 3), int(height / 2)), FONT, 2, COLOR, 2)

        for result in distance_results:
            if result['dangerous']:
                left, right = self.calc_x(result['detection1']['depth_x'])
                top, bottom = self.calc_z(result['detection1']['depth_z'])
                cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                left, right = self.calc_x(result['detection2']['depth_x'])
                top, bottom = self.calc_z(result['detection2']['depth_z'])
                cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 0, 255), 2)

        combined = np.hstack((frame, bird_frame))
        cv2.imshow("Frame", combined)

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()


    def calc_x(self, val):
        norm = min(MAX_X, max(val, MIN_X))
        center = (norm - MIN_X) / (MAX_X - MIN_X) * BIRD_FRAME_X
        bottom_x = max(center - 2, 0)
        top_x = min(center + 2, BIRD_FRAME_X)
        return int(bottom_x), int(top_x)


    def calc_z(self, val):
        norm = min(MAX_Z, max(val, MIN_Z))
        center = (1 - (norm - MIN_Z) / (MAX_Z - MIN_Z)) * BIRD_FRAME_Z
        bottom_z = max(center - 2, 0)
        top_z = min(center + 2, BIRD_FRAME_Z)
        return int(bottom_z), int(top_z)


    def make_bird_frame(self):
        fov = 68.7938
        min_distance = 0.827
        frame = np.zeros((BIRD_FRAME_Z, BIRD_FRAME_X, 3), np.uint8)
        min_y = int((1 - (min_distance - MIN_Z) / (MAX_Z - MIN_Z)) * frame.shape[0])
        cv2.rectangle(frame, (0, min_y), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, frame.shape[0]),
            (frame.shape[1], frame.shape[0]),
            (frame.shape[1], max_p),
            (center, frame.shape[0]),
            (0, max_p),
            (0, frame.shape[0]),
        ])
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame