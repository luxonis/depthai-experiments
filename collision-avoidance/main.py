import logging
import math

import cv2
import numpy as np

from config import DEBUG
from crash_avoidance import CrashAvoidance
from depthai_utils import DepthAI, DepthAIDebug
from tracker import Tracker

log = logging.getLogger(__name__)


class Main:
    depthai_class = DepthAI

    def __init__(self):
        self.depthai = self.depthai_class(model_name="vehicle-detection-adas-0002", threshold=0.5)
        self.tracker = Tracker()
        self.crash_avoidance = CrashAvoidance()

    def parse_frame(self, frame, results):
        pts = [(item.depth_x, item.depth_z) for item in results]
        tracker_objs = self.tracker.update(pts)
        crash_alert = self.crash_avoidance.parse(tracker_objs)
        if crash_alert:
            cv2.putText(frame, "DANGER!", (frame.shape[0] // 2, frame.shape[1] // 2 - 20), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 250))

    def run(self):
        try:
            log.info("Setup complete, parsing frames...")
            for frame, results in self.depthai.capture():
                self.parse_frame(frame, results)
        finally:
            del self.depthai


class MainDebug(Main):
    depthai_class = DepthAIDebug
    max_z = 15
    min_z = 0
    max_x = 1.3
    min_x = -0.5

    def __init__(self):
        super().__init__()
        self.distance_bird_frame = self.make_bird_frame()

    def make_bird_frame(self):
        fov = 68.7938
        min_distance = 0.827
        frame = np.zeros((384, 100, 3), np.uint8)
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

    def calc_x(self, val, padding=0):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.distance_bird_frame.shape[1]
        if padding:
            bottom_x = max(center - padding, 0)
            top_x = min(center + padding, self.distance_bird_frame.shape[1])
            return int(bottom_x), int(top_x)
        else:
            return int(center)

    def calc_z(self, val, padding=0):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.distance_bird_frame.shape[0]
        if padding:
            bottom_z = max(center - padding, 0)
            top_z = min(center + padding, self.distance_bird_frame.shape[0])
            return int(bottom_z), int(top_z)
        else:
            return int(center)

    def parse_frame(self, frame, results):
        super().parse_frame(frame, results)
        bird_frame = self.distance_bird_frame.copy()

        for result in results:
            left, right = self.calc_x(result.depth_x, padding=2)
            top, bottom = self.calc_z(result.depth_z, padding=2)
            cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        for key in list(self.tracker.history.keys()):
            points = self.tracker.history[key]
            color = self.tracker.colors[key]

            draw_pts = np.array([(self.calc_x(pt[0]), self.calc_z(pt[1])) for pt in points])
            cv2.polylines(bird_frame, [draw_pts], False, color=color)

        numpy_horizontal = np.hstack((frame, bird_frame))
        cv2.imshow("DetectionFrame", numpy_horizontal)
        
        cv2.moveWindow("Bird", 1050, 200)
        cv2.moveWindow("Frame", 1370, 200)
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
