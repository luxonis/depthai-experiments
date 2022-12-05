import logging
import math

import cv2
import numpy as np

from alerting import AlertingGate, AlertingGateDebug
from config import MODEL_NAME, DEBUG
from depthai_utils import DepthAI, DepthAIDebug
from distance import DistanceGuardian, DistanceGuardianDebug

log = logging.getLogger(__name__)


class Main:
    depthai_class = DepthAI
    distance_guardian_class = DistanceGuardian
    alerting_gate_class = AlertingGate

    def __init__(self):
        self.depthai = self.depthai_class(MODEL_NAME)
        self.distance_guardian = self.distance_guardian_class()
        self.alerting_gate = self.alerting_gate_class()

    def parse_frame(self, frame, results):
        distance_results = self.distance_guardian.parse_frame(frame, results)
        should_alert = self.alerting_gate.parse_frame(distance_results)
        if should_alert:
            img_h = frame.shape[0]
            img_w = frame.shape[1]
            cv2.putText(frame, "Too close", (int(img_w / 3), int(img_h / 2)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 1)
        return distance_results, should_alert

    def run(self):
        try:
            log.info("Setup complete, parsing frames...")
            for frame, results in self.depthai.capture():
                self.parse_frame(frame, results)
        finally:
            del self.depthai


class MainDebug(Main):
    depthai_class = DepthAIDebug
    distance_guardian_class = DistanceGuardianDebug
    alerting_gate_class = AlertingGateDebug
    max_z = 4
    min_z = 1
    max_x = 0.9
    min_x = -0.7

    def __init__(self):
        super().__init__()
        self.distance_bird_frame = self.make_bird_frame()

    def make_bird_frame(self):
        fov = 68.7938
        min_distance = 0.827
        frame = np.zeros((320, 100, 3), np.uint8)
        min_y = int((1 - (min_distance - self.min_z) / (self.max_z - self.min_z)) * frame.shape[0])
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

    def calc_x(self, val):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.distance_bird_frame.shape[1]
        bottom_x = max(center - 2, 0)
        top_x = min(center + 2, self.distance_bird_frame.shape[1])
        return int(bottom_x), int(top_x)

    def calc_z(self, val):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.distance_bird_frame.shape[0]
        bottom_z = max(center - 2, 0)
        top_z = min(center + 2, self.distance_bird_frame.shape[0])
        return int(bottom_z), int(top_z)

    def parse_frame(self, frame, results):
        distance_results, should_alert = super().parse_frame(frame, results)

        bird_frame = self.distance_bird_frame.copy()
        too_close_ids = []
        for result in distance_results:
            if result['dangerous']:
                left, right = self.calc_x(result['detection1']['depth_x'])
                top, bottom = self.calc_z(result['detection1']['depth_z'])
                cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                too_close_ids.append(result['detection1']['id'])
                left, right = self.calc_x(result['detection2']['depth_x'])
                top, bottom = self.calc_z(result['detection2']['depth_z'])
                cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                too_close_ids.append(result['detection2']['id'])

        for result in results:
            if result['id'] not in too_close_ids:
                left, right = self.calc_x(result['depth_x'])
                top, bottom = self.calc_z(result['depth_z'])
                cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        numpy_horizontal = np.hstack((frame, bird_frame))
        cv2.imshow("Frame", numpy_horizontal)
        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()


if __name__ == '__main__':
    if DEBUG:
        log.info("Setting up debug run...")
        MainDebug().run()
    else:
        log.info("Setting up non-debug run...")
        Main().run()
