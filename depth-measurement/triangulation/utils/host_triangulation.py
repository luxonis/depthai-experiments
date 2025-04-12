import cv2
import numpy as np
import depthai as dai
from depthai_nodes import (
    ImgDetectionsExtended,
    ImgDetectionExtended,
)
from typing import Tuple, List
from utility import TextHelper
from .annotation_helper import AnnotationHelper
from .stereo_inference import StereoInference


class Triangulation(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._leftColor = (1, 0, 0, 1)
        self._rightColor = (0, 1, 0, 1)
        self._textHelper = TextHelper()
        self.combined_frame = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.combined_keypoints = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.annot_left = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.annot_right = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

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

        bbox_annot_left = AnnotationHelper()
        for detection in nn_face_left.detections:
            rect = detection.rotated_rect
            x = int(rect.center.x * left_frame.shape[1])
            y = int(rect.center.y * left_frame.shape[0])
            w = int(rect.size.width * left_frame.shape[1])
            h = int(rect.size.height * left_frame.shape[0])
            top_left = (x - w/2, y - h/2)
            bottom_right = (x + w/2, y + h/2)
            
            bbox_annot_left.draw_rectangle(
                top_left=top_left,
                bottom_right=bottom_right,
                outline_color=self._leftColor,
                thickness=2
            )

        bbox_annot_left_msg = bbox_annot_left.build(
            timestamp=face_left.getTimestamp(), sequence_num=face_left.getSequenceNum()
        )
        self.annot_left.send(bbox_annot_left_msg)

        bbox_annot_right = AnnotationHelper()
        for detection in nn_face_right.detections:
            rect = detection.rotated_rect
            x = int(rect.center.x * right_frame.shape[1])
            y = int(rect.center.y * right_frame.shape[0])
            w = int(rect.size.width * right_frame.shape[1])
            h = int(rect.size.height * right_frame.shape[0])
            top_left = (x - w/2, y - h/2)
            bottom_right = (x + w/2, y + h/2)
            
            bbox_annot_right.draw_rectangle(
                top_left=top_left,
                bottom_right=bottom_right,
                outline_color=self._rightColor,
                thickness=2
            )

        bbox_annot_right_msg = bbox_annot_right.build(
            timestamp=face_right.getTimestamp(), sequence_num=face_right.getSequenceNum()
        )
        self.annot_right.send(bbox_annot_right_msg)

        combined = cv2.addWeighted(left_frame, 0.5, right_frame, 0.5, 0)
        y_dimension, x_dimension = combined.shape[:2]

        annotation_helper = AnnotationHelper()
        if nn_face_left.detections and nn_face_right.detections:
            spatials = []
            keypoints = zip(
                nn_face_left.detections[0].keypoints,
                nn_face_right.detections[0].keypoints,
            )
            
            for i, (keypoint_left, keypoint_right) in enumerate(keypoints):
                coords_left = (
                    int(keypoint_left.x * x_dimension),
                    int(keypoint_left.y * y_dimension),
                )
                coords_right = (
                    int(keypoint_right.x * x_dimension),
                    int(keypoint_right.y * y_dimension),
                )

                rel_coords_left = (
                    coords_left[0] / x_dimension,
                    coords_left[1] / y_dimension,
                )
                rel_coords_right = (
                    coords_right[0] / x_dimension,
                    coords_right[1] / y_dimension,
                )

                # Visualize keypoints
                annotation_helper.draw_circle(
                    center=rel_coords_left,
                    radius=3 / x_dimension,
                    outline_color=self._leftColor,
                    thickness=1,
                )
                annotation_helper.draw_circle(
                    center=rel_coords_right,
                    radius=3 / x_dimension, 
                    outline_color=self._rightColor,
                    thickness=1,
                )

                # Visualize disparity line
                annotation_helper.draw_line(
                    pt1=rel_coords_left,
                    pt2=rel_coords_right,
                    color=self._leftColor,
                    thickness=1,
                )

                # Calculate spatial data
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
                        annotation_helper.draw_text(
                            text=s,
                            position=(10, y),
                            color=(1.0, 1.0, 1.0, 1.0),
                            background_color=(0.0, 0.0, 0.0, 0.7),
                            size=6
                        )
                        y += y_delta

        output_frame = self._create_output_frame(face_left, combined)

        self.combined_frame.send(output_frame)

        annotations_msg = annotation_helper.build(
            timestamp=face_left.getTimestamp(), sequence_num=face_left.getSequenceNum()
        )
        self.combined_keypoints.send(annotations_msg)

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
