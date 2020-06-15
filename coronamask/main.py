import logging
import cv2

from depthai_utils import DepthAI, DepthAIDebug
from config import MODEL_LOCATION, DEBUG

log = logging.getLogger(__name__)


class Main:
    depthai_class = DepthAI

    def __init__(self):
        self.depthai = self.depthai_class(MODEL_LOCATION, 'people')

    def parse_frame(self, frame, results):
        pass

    def run(self):
        try:
            log.info("Setup complete, parsing frames...")
            for frame, results in self.depthai.capture():
                self.parse_frame(frame, results)
        finally:
            del self.depthai


class MainDebug(Main):
    depthai_class = DepthAIDebug

    def parse_frame(self, frame, results):
        super().parse_frame(frame, results)

        cv2.imshow("Frame", frame)
        cv2.moveWindow("Frame", 1100, 200)
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
