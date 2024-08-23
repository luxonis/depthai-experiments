import depthai as dai
import cv2
import time

from enum import Enum


class FPSDrawer(dai.node.HostNode):
    class Position(Enum):
        TOP_LEFT, \
        BOTTOM_LEFT, \
        TOP_RIGHT, \
        BOTTOM_RIGHT = range(4)

    def __init__(self) -> None:
        super().__init__()

        self._start_time = time.monotonic()
        self._frame_count = 0
        self.position = FPSDrawer.Position.BOTTOM_RIGHT

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, preview: dai.Node.Output) -> "FPSDrawer":
        self.link_args(preview)
        return self

    def process(self, preview: dai.ImgFrame) -> None:
        frame = preview.getCvFrame()
        self._frame_count += 1

        if self.position == FPSDrawer.Position.TOP_LEFT:
            position = 4, 17
        elif self.position == FPSDrawer.Position.BOTTOM_LEFT:
            position = 4, max(0, frame.shape[0] - 10)
        elif self.position == FPSDrawer.Position.TOP_RIGHT:
            position = max(0, frame.shape[1] - 125), 17
        else:
            position = max(0, frame.shape[1] - 125), max(0, frame.shape[0] - 10)

        fps = self._frame_count / (time.monotonic() - self._start_time)
        cv2.putText(frame, f"NN fps: {fps:.2f}", position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(255, 255, 255))

        preview.setCvFrame(frame, preview.getType())
        self.output.send(preview)

    def setFpsPosition(self, position: "FPSDrawer.Position") -> None:
        self.position = position
