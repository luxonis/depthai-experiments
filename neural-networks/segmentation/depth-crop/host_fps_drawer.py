import depthai as dai
import cv2
import time

from enum import Enum


class FPSDrawer(dai.node.HostNode):
    class Position(Enum):
        TOP_LEFT, BOTTOM_LEFT, TOP_RIGHT, BOTTOM_RIGHT = range(4)

    def __init__(self) -> None:
        super().__init__()
        self._frames = []

        self.remove_old_frames = True
        self.position = FPSDrawer.Position.BOTTOM_LEFT

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(self, preview: dai.Node.Output) -> "FPSDrawer":
        self.link_args(preview)
        return self

    def process(self, preview: dai.Buffer) -> None:
        assert isinstance(preview, dai.ImgFrame)

        frame = preview.getCvFrame()
        now = time.monotonic()

        if self.remove_old_frames:
            self._remove_old_frames(now)
        self._frames.append(now)

        if self.position == FPSDrawer.Position.TOP_LEFT:
            position = 4, 17
        elif self.position == FPSDrawer.Position.BOTTOM_LEFT:
            position = 4, max(0, frame.shape[0] - 10)
        elif self.position == FPSDrawer.Position.TOP_RIGHT:
            position = max(0, frame.shape[1] - 125), 17
        else:
            position = max(0, frame.shape[1] - 125), max(0, frame.shape[0] - 10)

        cv2.putText(
            frame,
            f"NN fps: {self._get_fps():.2f}",
            position,
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            color=(255, 255, 255),
        )

        output_frame = dai.ImgFrame()
        output_frame.setCvFrame(frame, preview.getType())
        output_frame.setTimestamp(preview.getTimestamp())
        output_frame.setTimestampDevice(preview.getTimestampDevice())
        output_frame.setInstanceNum(preview.getInstanceNum())
        self.output.send(output_frame)

    def setFpsPosition(self, position: "FPSDrawer.Position") -> None:
        self.position = position

    def setRemoveOldFrames(self, remove: bool) -> None:
        self.remove_old_frames = remove

    def _remove_old_frames(self, now: float) -> None:
        while self._frames and (now - self._frames[0]) > 1:
            self._frames.pop(0)

    def _get_fps(self) -> int:
        if len(self._frames) < 2:
            return 0
        return len(self._frames) / (self._frames[-1] - self._frames[0])
