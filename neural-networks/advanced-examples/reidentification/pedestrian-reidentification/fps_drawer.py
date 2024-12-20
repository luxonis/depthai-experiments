import time
import cv2
import depthai as dai
import numpy as np


class FPSDrawer(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input = self.createInput(types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self._last_frames = []


    def run(self) -> None:
        while self.isRunning():
            img_frame: dai.ImgFrame = self.input.get()
            now = time.time()
            self._remove_old_frames(now)
            self._last_frames.append(now)
            
            frame = img_frame.getCvFrame()
            fps = self._get_fps()
            self._draw_fps(frame, fps)

            frame_retyped = dai.ImgFrame()
            frame_retyped.setCvFrame(img_frame.getCvFrame(), dai.ImgFrame.Type.BGR888p)

            self.output.send(frame_retyped)


    def _remove_old_frames(self, now: float) -> None:
        while self._last_frames and (now - self._last_frames[0]) > 1:
            self._last_frames.pop(0)


    def _get_fps(self) -> int:
        return len(self._last_frames)


    def _draw_fps(self, frame: np.ndarray, fps: int) -> None:
        txt = f"FPS: {fps}"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
    