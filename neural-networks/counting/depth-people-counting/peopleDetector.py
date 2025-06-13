import depthai as dai
import numpy as np
import cv2
from utility import DETECTION_ROI
from typing import Sequence, Tuple, List


class PeopleDetector(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.disparity_multiplier = None
        self.out_depth_rgb = self.createOutput()
        self.out_debug_contours = self.createOutput()

    def build(
        self,
        depth: dai.Node.Output,
        disparity_multiplier: float,
    ) -> "PeopleDetector":
        self.disparity_multiplier = disparity_multiplier

        self.link_args(depth)
        return self

    def process(self, depth: dai.ImgFrame) -> None:
        depth_frame = depth.getCvFrame()
        depth_frame = (depth_frame * (self.disparity_multiplier)).astype(np.uint8)
        depth_rgb = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

        contours = self.get_contours(depth_frame)
        dets = self.get_detections(contours, depth_rgb)
        dets.setSequenceNum(depth.getSequenceNum())
        dets.setTimestamp(depth.getTimestamp())
        dets.setTransformation(depth.getTransformation())

        self.out.send(dets)
        depth_rgb_frame = dai.ImgFrame()
        depth_rgb_frame.setCvFrame(depth_rgb, dai.ImgFrame.Type.BGR888p)
        self.out_depth_rgb.send(depth_rgb_frame)

    def frame_norm(self, frame, bbox) -> Tuple[float, float, float, float]:
        width, height = (
            # frame.shape[0] + DETECTION_ROI[0],
            # frame.shape[1] + DETECTION_ROI[1],
            frame.shape[1],
            frame.shape[0],
        )
        return bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height

    def get_detections(self, contours, depth_rgb) -> dai.ImgDetections:
        dets = dai.ImgDetections()
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            # x += depth_rgb.shape[0] - DETECTION_ROI[0]
            # y += depth_rgb.shape[1] - DETECTION_ROI[1]
            area = w * h

            if 1000 < area:
                # Send the detection to the device - ObjectTracker node
                det = dai.ImgDetection()
                det.label = 0
                det.confidence = 1.0
                det.xmin, det.ymin, det.xmax, det.ymax = self.frame_norm(
                    depth_rgb, (x, y, x + w, y + h)
                )
                print(det.xmin, det.ymin, det.xmax, det.ymax)
                dets.detections = [det]

                # Draw rectangle on the biggest countour
                # self.text.rectangle(depth_rgb, (x, y), (x + w, y + h), size=2.5)
        return dets

    def get_contours(self, depth_frame) -> Sequence[np.ndarray]:
        cropped = depth_frame[
            DETECTION_ROI[1] : DETECTION_ROI[3], DETECTION_ROI[0] : DETECTION_ROI[2]
        ]

        _, thresh = cv2.threshold(cropped, 100, 145, cv2.THRESH_BINARY)
        blob = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)),
        )
        edged = cv2.Canny(blob, 20, 80)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        offset_x = DETECTION_ROI[0]
        offset_y = DETECTION_ROI[1]

        # edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
        # depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            contour[:, 0, 0] += offset_x
            contour[:, 0, 1] += offset_y

        # debug = dai.ImgFrame()
        # debug.setCvFrame(depth_frame, dai.ImgFrame.Type.BGR888p)
        # self.out_debug.send(debug)

        return contours

    def to_planar(self, arr: np.ndarray) -> List:
        return arr.transpose(2, 0, 1).flatten()
