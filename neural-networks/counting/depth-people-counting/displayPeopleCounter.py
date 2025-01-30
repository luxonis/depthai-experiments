import depthai as dai
import numpy as np
import cv2
from utility import TextHelper, PeopleCounter, DETECTION_ROI
from typing import Sequence


class DisplayPeopleCounter(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.counter = PeopleCounter()
        self.text = TextHelper()

    def build(
        self,
        depth_in: dai.Node.Output,
        tracklets_in: dai.Node.Output,
        det_in_q: dai.InputQueue,
        frame_in_q: dai.InputQueue,
        disparity_multiplier: float,
    ) -> "DisplayPeopleCounter":
        self.disparity_multiplier = disparity_multiplier

        self.det_in_q = det_in_q
        self.frame_in_q = frame_in_q
        self.tracklets_out = tracklets_in.createOutputQueue()

        self.link_args(depth_in)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, depth_out: dai.ImgFrame) -> None:
        depth_frame = depth_out.getCvFrame()
        depth_frame = (depth_frame * (self.disparity_multiplier)).astype(np.uint8)
        depth_rgb = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

        tracklets_in = self.tracklets_out.tryGet()
        if tracklets_in is not None:
            self.counter.new_tracklets(tracklets_in.tracklets)

        contours = self.get_contours(depth_frame)
        dets = self.get_detections(contours, depth_rgb)

        self.det_in_q.send(dets)

        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(depth_rgb, dai.ImgFrame.Type.BGR888p)  # works

        self.frame_in_q.send(img_frame)

        self.text.rectangle(
            depth_rgb,
            (DETECTION_ROI[0], DETECTION_ROI[1]),
            (DETECTION_ROI[2], DETECTION_ROI[3]),
        )
        self.text.putText(depth_rgb, str(self.counter), (20, 40))

        cv2.imshow("depth", depth_rgb)

        if cv2.waitKey(1) == ord("q"):
            self.stopPipeline()

    def frame_norm(self, frame, bbox) -> tuple[float, float, float, float]:
        width, height = (
            frame.shape[0] + DETECTION_ROI[0],
            frame.shape[1] + DETECTION_ROI[1],
        )
        return bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height

    def get_detections(self, contours, depth_rgb) -> dai.ImgDetections:
        dets = dai.ImgDetections()
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            x += DETECTION_ROI[0]
            y += DETECTION_ROI[1]
            area = w * h

            if 15000 < area:
                # Send the detection to the device - ObjectTracker node
                det = dai.ImgDetection()
                det.label = 1
                det.confidence = 1.0
                det.xmin, det.ymin, det.xmax, det.ymax = self.frame_norm(
                    depth_rgb, (x, y, x + w, y + h)
                )
                dets.detections = [det]

                # Draw rectangle on the biggest countour
                self.text.rectangle(depth_rgb, (x, y), (x + w, y + h), size=2.5)
        return dets

    def get_contours(self, depth_frame) -> Sequence[np.ndarray]:
        cropped = depth_frame[
            DETECTION_ROI[1] : DETECTION_ROI[3], DETECTION_ROI[0] : DETECTION_ROI[2]
        ]
        _, thresh = cv2.threshold(cropped, 125, 145, cv2.THRESH_BINARY)
        blob = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)),
        )
        edged = cv2.Canny(blob, 20, 80)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def to_planar(self, arr: np.ndarray) -> list:
        return arr.transpose(2, 0, 1).flatten()
