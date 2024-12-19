from datetime import timedelta
from enum import Enum, auto

import cv2
import depthai as dai
import numpy as np


class DrawTracklet(dai.node.HostNode):
    class Position(Enum):
        TOP_LEFT = auto()
        BOTTOM_LEFT = auto()
        TOP_RIGHT = auto()
        BOTTOM_RIGHT = auto()
        CENTER = auto()

    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.color = (255, 0, 0)
        self.thickness = 2
        self.padding = 5

    def build(
        self,
        frame: dai.Node.Output,
        tracklets: dai.Node.Output,
        text_position: Position = Position.CENTER,
    ) -> "DrawTracklet":
        self._text_position = text_position
        self.link_args(frame, tracklets)

        return self

    def process(self, frame: dai.ImgFrame, tracklets: dai.Tracklets) -> None:
        img = frame.getCvFrame().copy()

        out_img = self._draw_tracklets(img, tracklets)
        img_frame = self._create_img_frame(
            out_img,
            dai.ImgFrame.Type.BGR888p,
            frame.getTimestamp(),
            frame.getSequenceNum(),
        )

        self.output.send(img_frame)

    def _draw_tracklets(self, frame: np.ndarray, tracklets: dai.Tracklets):
        for t in tracklets.tracklets:
            frame = self._draw_tracklet_id(frame, t)
            top_left = (int(t.roi.topLeft().x), int(t.roi.topLeft().y))
            bottom_right = (int(t.roi.bottomRight().x), int(t.roi.bottomRight().y))
            cv2.rectangle(frame, top_left, bottom_right, self.color, self.thickness)
        return frame

    def _draw_tracklet_id(self, frame, tracklet: dai.Tracklet):
        text = f"ID: {tracklet.id}"

        text_w, text_h = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.thickness
        )[0]
        if self._text_position == DrawTracklet.Position.TOP_LEFT:
            point = (
                int(tracklet.roi.topLeft().x + self.padding),
                int(tracklet.roi.topLeft().y + self.padding + text_h),
            )
        elif self._text_position == DrawTracklet.Position.BOTTOM_LEFT:
            point = (
                int(tracklet.roi.topLeft().x + self.padding),
                int(tracklet.roi.bottomRight().y - self.padding),
            )
        elif self._text_position == DrawTracklet.Position.TOP_RIGHT:
            point = (
                int(tracklet.roi.bottomRight().x - self.padding - text_w),
                int(tracklet.roi.topLeft().y + self.padding + text_h),
            )
        elif self._text_position == DrawTracklet.Position.BOTTOM_RIGHT:
            point = (
                int(tracklet.roi.bottomRight().x - self.padding - text_w),
                int(tracklet.roi.bottomRight().y - self.padding),
            )
        else:
            point = (
                int((tracklet.roi.topLeft().x + tracklet.roi.bottomRight().x) / 2),
                int((tracklet.roi.topLeft().y + tracklet.roi.bottomRight().y) / 2),
            )
        cv2.putText(
            frame,
            text,
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color=self.color,
            thickness=self.thickness,
        )
        return frame

    def _create_img_frame(
        self,
        frame: np.ndarray,
        type: dai.ImgFrame.Type,
        timestamp: timedelta,
        sequence_num: int,
    ) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(frame, type)
        img_frame.setTimestamp(timestamp)
        img_frame.setSequenceNum(sequence_num)
        return img_frame

    def set_color(self, color: tuple[int, int, int]) -> None:
        self.color = color

    def set_thickness(self, thickness: int) -> None:
        self.thickness = thickness

    def set_padding(self, padding: int) -> None:
        self.padding = padding

    def set_text_position(self, position: Position) -> None:
        self._text_position = position
