import cv2
import depthai as dai
import numpy as np


class BlurBboxes(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.rounded_blur = False
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self, frame: dai.Node.Output, nn: dai.Node.Output, rounded_blur: bool = False
    ) -> "BlurBboxes":
        self.rounded_blur = rounded_blur
        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, in_frame: dai.ImgFrame, in_detections: dai.Buffer) -> None:
        frame = in_frame.getCvFrame()
        assert isinstance(in_detections, dai.ImgDetections)
        out_frame = self.blur_detections(frame, in_detections.detections)

        img = self._create_img_frame(out_frame, dai.ImgFrame.Type.BGR888p)

        self.output.send(img)

    def blur_detections(
        self, frame: np.ndarray, detections: list[dai.ImgDetection]
    ) -> np.ndarray:
        frame_copy = frame.copy()
        for detection in detections:
            bbox = [
                int(detection.xmin),
                int(detection.ymin),
                int(detection.xmax),
                int(detection.ymax),
            ]

            roi = frame_copy[bbox[1] : bbox[3], bbox[0] : bbox[2]]

            roi_width = bbox[2] - bbox[0]
            roi_height = bbox[3] - bbox[1]

            if self.rounded_blur:
                mask = np.zeros((roi_height, roi_width), np.uint8)
                polygon = cv2.ellipse2Poly(
                    (int(roi_width / 2), int(roi_height / 2)),
                    (int(roi_width / 2), int(roi_height / 2)),
                    0,
                    0,
                    360,
                    delta=1,
                )
                cv2.fillConvexPoly(mask, polygon, 255)
            else:
                mask = np.full((roi_height, roi_width), 255, np.uint8)

            blurred_roi = cv2.blur(roi, (80, 80))

            blurred_ellipse = cv2.bitwise_and(blurred_roi, blurred_roi, mask=mask)
            inverse_mask = cv2.bitwise_not(mask)
            original_background = cv2.bitwise_and(roi, roi, mask=inverse_mask)
            combined = cv2.add(blurred_ellipse, original_background)

            frame_copy[bbox[1] : bbox[3], bbox[0] : bbox[2]] = combined
        return frame_copy

    def set_use_ellipses(self, use_ellipses: bool):
        self.rounded_blur = use_ellipses

    def _create_img_frame(
        self, frame: np.ndarray, type: dai.ImgFrame.Type
    ) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(frame, type)
        return img_frame
