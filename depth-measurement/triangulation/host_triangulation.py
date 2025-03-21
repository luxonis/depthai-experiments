import cv2
import numpy as np
import depthai as dai
from depthai_nodes import (
    ImgDetectionsExtended,
    ImgDetectionExtended,
)
from typing import Tuple, List
from utility import TextHelper
from stereo_inference import StereoInference


class Triangulation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._leftColor = (255, 0, 0)
        self._rightColor = (0, 255, 0)
        self._textHelper = TextHelper()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self._display_detections = True
        self.sendProcessingToPipeline(True)

    def build(
        self,
        face_left: dai.Node.Output,
        face_right: dai.Node.Output,
        face_nn_left: dai.Node.Output,
        face_nn_right: dai.Node.Output,
        device: dai.Device,
        resolution_number: Tuple[int, int],
    ) -> "Triangulation":
        self.link_args(face_left, face_right, face_nn_left, face_nn_right)
        self._stereoInference = StereoInference(device, resolution_number)

        return self

    def set_display_detections(self, display_detections: bool) -> None:
        self._display_detections = display_detections

    def process(
        self,
        face_left: dai.ImgFrame,
        face_right: dai.ImgFrame,
        nn_face_left: dai.Buffer,
        nn_face_right: dai.Buffer,
    ) -> None:
        assert isinstance(nn_face_left, ImgDetectionsExtended)
        assert isinstance(nn_face_right, ImgDetectionsExtended)

        left_frame = face_left.getCvFrame()
        right_frame = face_right.getCvFrame()

        if self._display_detections:
            self._displayDetections(
                left_frame, nn_face_left.detections, self._leftColor
            )
            self._displayDetections(
                right_frame, nn_face_right.detections, self._rightColor
            )

        combined = cv2.addWeighted(left_frame, 0.5, right_frame, 0.5, 0)

        if nn_face_left.detections and nn_face_right.detections:
            spatials = []
            keypoints = zip(
                nn_face_left.detections[0].keypoints,
                nn_face_right.detections[0].keypoints,
            )
            y_dimension, x_dimension = left_frame.shape[:2]
            for i, (keypoint_left, keypoint_right) in enumerate(keypoints):
                coords_left = (
                    int(keypoint_left.x * x_dimension),
                    int(keypoint_left.y * y_dimension),
                )
                coords_right = (
                    int(keypoint_right.x * x_dimension),
                    int(keypoint_right.y * y_dimension),
                )

                self._draw_keypoint(left_frame, coords_left, self._leftColor)
                self._draw_keypoint(right_frame, coords_right, self._rightColor)
                self._draw_keypoint(combined, coords_left, self._leftColor)
                self._draw_keypoint(combined, coords_right, self._rightColor)

                # Visualize disparity line frame
                self._draw_disparity_line(combined, coords_left, coords_right)

                disparity = self._stereoInference.calculate_distance(
                    coords_left, coords_right
                )
                depth = self._stereoInference.calculate_depth(disparity)
                spatial = self._stereoInference.calc_spatials(coords_right, depth)
                spatials.append(spatial)

                if i == 0:
                    y = 0
                    y_delta = 18
                    strings = [
                        "Disparity: {:.0f} pixels".format(disparity),
                        "X: {:.2f} m".format(spatial[0] / 1000),
                        "Y: {:.2f} m".format(spatial[1] / 1000),
                        "Z: {:.2f} m".format(spatial[2] / 1000),
                    ]
                    for s in strings:
                        y += y_delta
                        self._textHelper.putText(combined, s, (10, y))

        combined = np.concatenate((left_frame, combined, right_frame), axis=1)
        output_frame = self._create_output_frame(face_left, combined)
        self.output.send(output_frame)

    def _draw_disparity_line(
        self,
        combined: np.ndarray,
        coords_left: Tuple[int, int],
        coords_right: Tuple[int, int],
    ) -> None:
        cv2.line(combined, coords_right, coords_left, (0, 0, 255), 1)

    def _draw_keypoint(
        self,
        left_frame: np.ndarray,
        coords_left: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        cv2.circle(left_frame, coords_left, 3, color)

    def _create_output_frame(
        self, face_left: dai.ImgFrame, combined: np.ndarray
    ) -> dai.ImgFrame:
        output_frame = dai.ImgFrame()
        output_frame.setCvFrame(combined, dai.ImgFrame.Type.BGR888i)
        output_frame.setTimestamp(face_left.getTimestamp())
        return output_frame

    def _displayDetections(
        self, frame: np.ndarray, detections: List[ImgDetectionExtended], color
    ) -> None:
        for detection in detections:
            rect = detection.rotated_rect
            x = int(rect.center.x * frame.shape[1])
            y = int(rect.center.y * frame.shape[0])
            w = int(rect.size.width * frame.shape[1])
            h = int(rect.size.height * frame.shape[0])
            cv2.rectangle(
                frame,
                (x - (w // 2), y - (h // 2)),
                ((x + (w // 2), y + (h // 2))),
                color,
                2,
            )
