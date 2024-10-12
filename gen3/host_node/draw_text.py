import time
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np


class TextConfig(dai.Buffer):
    def __init__(self) -> None:
        super().__init__(0)
        self._position = (0, 0)
        self._size = 0.5
        self._color = (255, 255, 255)
        self._thickness = 2

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._position = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: float) -> None:
        self._size = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        self._color = value

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value: int) -> None:
        self._thickness = value


class TextMessage(dai.Buffer):
    def __init__(self, text: str) -> None:
        super().__init__(0)
        self._text = text

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value


class DrawText(dai.node.ThreadedHostNode):
    FPS_TOLERANCE_DIVISOR = 2
    INPUT_CHECKS_PER_FPS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self) -> None:
        super().__init__()
        self._camera_fps = 30

        self._current_config = TextConfig()
        self._current_text = TextMessage("")

        self.input_frame = dai.Node.Input(
            self, types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)]
        )
        self.input_config = dai.Node.Input(
            self, types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)]
        )
        self.input_text = dai.Node.Input(
            self, types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)]
        )

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def set_camera_fps(self, fps: int) -> None:
        self._camera_fps = fps

    def build(
        self,
        frame: dai.Node.Output,
        text: dai.Node.Output,
        config: dai.Node.Output | None = None,
    ) -> "DrawText":
        frame.link(self.input_frame)
        text.link(self.input_text)
        if config:
            config.link(self.input_config)

        return self

    def run(self) -> None:
        while self.isRunning():
            config_msg = self.input_config.tryGet()
            if config_msg is not None:
                assert isinstance(config_msg, TextConfig)
                self._current_config = config_msg

            text_msg = self.input_text.tryGet()
            if text_msg is not None:
                assert isinstance(text_msg, TextMessage)
                self._current_text = text_msg

            frame_msg = self.input_frame.tryGet()
            if frame_msg is not None:
                assert isinstance(frame_msg, dai.ImgFrame)
                img = frame_msg.getCvFrame().copy()
                img_text = self._draw_text(img)
                img_frame = self._create_img_frame(
                    img_text, frame_msg.getTimestamp(), frame_msg.getSequenceNum()
                )
                self.output.send(img_frame)

            time.sleep(1 / self.INPUT_CHECKS_PER_FPS / self._camera_fps)

    def _create_img_frame(
        self, img: np.ndarray, timestamp: timedelta, sequence_num: int
    ) -> dai.ImgFrame:
        frame = dai.ImgFrame()
        frame.setCvFrame(img, dai.ImgFrame.Type.BGR888p)
        frame.setTimestamp(timestamp)
        frame.setSequenceNum(sequence_num)
        return frame

    def _draw_text(self, frame: np.ndarray):
        lines = self._current_text.text.split("\n")
        size, _ = cv2.getTextSize(
            self._current_text.text,
            self.FONT,
            self._current_config.size,
            self._current_config.thickness,
        )
        _, text_h = size
        for i, line in enumerate(lines):
            y = self._current_config.position[1] + int(i * text_h * 1.5)
            cv2.putText(
                frame,
                line,
                (self._current_config.position[0], y),
                self.FONT,
                self._current_config.size,
                self._current_config.color,
                self._current_config.thickness,
            )
        return frame

    @property
    def config(self) -> TextConfig:
        return self._current_config
