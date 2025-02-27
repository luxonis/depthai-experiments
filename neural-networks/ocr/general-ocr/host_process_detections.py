import depthai as dai

from utils import RotatedRectBuffer
import depthai_nodes
import cv2
import numpy as np


class ProcessDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.passthrough = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.output_rect = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )
        self.output_config = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfig, True)
            ]
        )
        self.display = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self, frame: dai.Node.Output, detections: dai.Node.Output
    ) -> "ProcessDetections":
        self.link_args(frame, detections)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, frame: dai.ImgFrame, detections: dai.Buffer) -> None:
        # Casting values that were sent in this exact format

        assert isinstance(
            detections, depthai_nodes.ml.messages.img_detections.CornerDetections
        )
        # print(detections.detections)

        key_pressed = cv2.waitKey(1)

        frame_with_bboxes = frame.getCvFrame()
        for idx, det in enumerate(detections.detections):
            corners = det.keypoints
            polygon = [
                [corners[0].x * 4, corners[0].y * 4],
                [corners[1].x * 4, corners[1].y * 4],
                [corners[2].x * 4, corners[2].y * 4],
                [corners[3].x * 4, corners[3].y * 4],
            ]

            rr = get_rotated_rect_from_detection(polygon)

            frame_with_bboxes = print_bboxes_on_frame(frame_with_bboxes, polygon)

            if key_pressed != ord("c"):
                continue

            # print(det)
            # print(rr.center.x, rr.center.y, rr.size.width, rr.size.height)
            # print("bačov pes")

            cfg = dai.ImageManipConfig()
            cfg.setFrameType(dai.ImgFrame.Type.BGR888p)
            cfg.setCropRotatedRect(rr, False)
            # cfg.setResizeThumbnail(320, 48)
            cfg.setResize(320, 48)
            cfg.setTimestamp(frame.getTimestamp())
            cfg.setTimestampDevice(frame.getTimestampDevice())

            rr_buffer = RotatedRectBuffer()
            rr_buffer.set_rect(rr)
            rr_buffer.setTimestamp(frame.getTimestamp())
            rr_buffer.setTimestampDevice(frame.getTimestampDevice())

            # Send outputs to device
            if idx == 0:
                self.passthrough.send(frame)
                cfg.setReusePreviousImage(False)
            else:
                cfg.setReusePreviousImage(True)
            self.output_rect.send(rr_buffer)
            self.output_config.send(cfg)

        output_frame = dai.ImgFrame()
        output_frame.setType(dai.ImgFrame.Type.BGR888i)
        output_frame.setFrame(frame_with_bboxes)
        output_frame.setWidth(frame.getWidth())
        output_frame.setHeight(frame.getHeight())
        output_frame.setTimestamp(frame.getTimestamp())
        output_frame.setTimestampDevice(frame.getTimestampDevice())
        self.display.send(output_frame)


def get_rotated_rect_from_detection(polygon: list[list[int]]) -> dai.RotatedRect:
    rr = dai.RotatedRect()

    rr.center.x = (polygon[0][0] + polygon[2][0]) // 2
    rr.center.y = (polygon[0][1] + polygon[2][1]) // 2

    rr.size.width = (
        (polygon[1][0] - polygon[0][0]) ** 2 + (polygon[1][1] - polygon[0][1]) ** 2
    ) ** (1 / 2)
    rr.size.height = (
        (polygon[2][0] - polygon[1][0]) ** 2 + (polygon[2][1] - polygon[1][1]) ** 2
    ) ** (1 / 2)

    rr.angle = np.arcsin((polygon[1][1] - polygon[0][1]) / rr.size.width) / np.pi * 180

    return rr


def print_bboxes_on_frame(
    frame: dai.ImgFrame, polygon: list[list[int]]
) -> dai.ImgFrame:
    polygon = np.array(polygon).astype(np.int32)
    cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

    return frame
