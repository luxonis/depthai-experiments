import depthai as dai
import cv2

class Display(dai.node.HostNode):
    # Only one instance of this class can call cv2.waitKey at a time
    _wait_key_instance = None

    def __init__(self) -> None:
        super().__init__()
        self.name = "Display"
        self.wait_for_exit = True
        self.process_wait_key = False

        if Display._wait_key_instance is None:
            self.process_wait_key = True
            Display._wait_key_instance = self

    def build(self, frame):
        self.sendProcessingToPipeline(True)
        self.link_args(frame)
        return self

    def setName(self, name: str) -> None:
        self.name = name

    def setWaitForExit(self, wait: bool) -> None:
        self.wait_for_exit = wait

    def process(self, frame) -> None:
        cv2.imshow(self.name, frame.getCvFrame())

        if self.process_wait_key and cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
