import logging

import cv2

from alerting import AlertingGate, AlertingGateDebug
from config import MODEL_LOCATION, DEBUG
from depthai_utils import DepthAI, DepthAIDebug
from distance import DistanceGuardian, DistanceGuardianDebug

log = logging.getLogger(__name__)


class Main:
    depthai_class = DepthAI
    distance_guardian_class = DistanceGuardian
    alerting_gate_class = AlertingGate

    def __init__(self):
        self.depthai = self.depthai_class(MODEL_LOCATION, 'people')
        self.distance_guardian = self.distance_guardian_class()
        self.alerting_gate = self.alerting_gate_class()

    def parse_frame(self, frame, results):
        distance_results = self.distance_guardian.parse_frame(frame, results)
        should_alert = self.alerting_gate.parse_frame(distance_results)
        if should_alert:
            img_h = frame.shape[0]
            img_w = frame.shape[1]
            cv2.putText(frame, "Too close", (int(img_w / 3), int(img_h / 2)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 1)

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

    def parse_frame(self, frame, results):
        super().parse_frame(frame, results)

        cv2.imshow("Frame", frame)
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
