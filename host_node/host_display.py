import depthai as dai
import cv2
from typing import List

try:
    from .keyboard_reader import KeyboardPress
except Exception as _:
    print("coudn't import keyboard_reader")


class Display(dai.node.HostNode):
    # Only one instance of this class can call cv2.waitKey at a time
    _wait_key_instance = None

    def __init__(self) -> None:
        super().__init__()
        self.name = "Display"
        self.process_wait_key = False
        self.keyboard_input_q = None

        if Display._wait_key_instance is None:
            self.process_wait_key = True
            Display._wait_key_instance = self

    def build(self, frames: dai.Node.Output) -> "Display":
        self.sendProcessingToPipeline(True)
        self.link_args(frames)
        return self

    def setName(self, name: str) -> None:
        self.name = name

    def setWaitForExit(self, wait: bool) -> None:
        if Display._wait_key_instance is None and wait:
            self.process_wait_key = True
            Display._wait_key_instance = self
        elif Display._wait_key_instance is self and not wait:
            self.process_wait_key = False
            Display._wait_key_instance = None

    def setKeyboardInput(self, keyboard_input: dai.Node.Output) -> None:
        if not self.process_wait_key:
            raise RuntimeError(
                "Keyboard input can only be set if Display is set to wait for exit"
            )
        self.keyboard_input_q = keyboard_input.createOutputQueue()

    def process(self, frame: dai.ImgFrame) -> None:
        cv2.imshow(self.name, frame.getCvFrame())

        if self.process_wait_key and ord("q") in self._waitKeys():
            self.stopPipeline()

    def _waitKeys(self) -> List[int]:
        if self.keyboard_input_q:
            key_presses: List[KeyboardPress] = self.keyboard_input_q.tryGetAll()
            if key_presses:
                return [key_press.key for key_press in key_presses]
            else:
                return [-1]
        else:
            return [cv2.waitKey(1)]
