import depthai as dai
import cv2
import threading


class Display(dai.node.HostNode):
    _lock = threading.Lock()
    _wait_key_instance = None

    def __init__(self) -> None:
        super().__init__()
        self.name = "Display"
        self.wait_for_exit = True

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

        self.process_wait_key()

    def process_wait_key(self):
        # Makes sure that the wait key is only processed once
        if self.wait_for_exit:
            with self._lock:
                if Display._wait_key_instance is None:
                    Display._wait_key_instance = self

                    try:
                        if cv2.waitKey(1) == ord('q'):
                            print("Pipeline exited.")
                            self.stopPipeline()
                    finally:
                        Display._wait_key_instance = None

