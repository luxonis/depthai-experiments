import numpy as np
import depthai as dai
from .box_estimator import BoxEstimator
from .img_annotation_helper import AnnotationHelper


class BoxMeasurement(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()

        self.color_input = self.createInput()
        self.pcl_input = self.createInput()

        self.passthrough = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.annotation_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.measurements_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.box_estimator = None
        self.intrinsics = None
        self.dist_coeffs = None
        self.min_box_size = None

    def build(
        self,
        color: dai.Node.Output,
        pcl: dai.Node.Output,
        cam_intrinsics: list,
        dist_coeffs: np.ndarray,
        max_dist: float,
        min_box_size: float,
    ) -> "BoxMeasurement":
        self.intrinsics = cam_intrinsics
        self.dist_coeffs = dist_coeffs
        self.min_box_size = min_box_size

        self.box_estimator = BoxEstimator(max_dist)
        color.link(self.color_input)
        pcl.link(self.pcl_input)
        return self

    def create_text_annot(self, text, pos):
        txt_annot = dai.TextAnnotation()
        txt_annot.fontSize = 10
        txt_annot.backgroundColor = dai.Color(0, 1, 0, 1)
        txt_annot.textColor = dai.Color(1, 1, 1, 1)
        txt_annot.position = dai.Point2f(*pos)
        txt_annot.text = text
        return txt_annot

    def run(self) -> None:
        while self.isRunning():
            color_msg = self.color_input.get()
            pcl_msg = self.pcl_input.get()
            color_frame = color_msg.getCvFrame()

            l, w, h = self.box_estimator.process_pcl(pcl_msg)

            bbox_annot_builder = AnnotationHelper()
            measurement_annot_builder = AnnotationHelper()

            if l * w * h > self.min_box_size:
                # Create ImgAnnotations and draw lines
                height, width, _ = color_frame.shape
                self.box_estimator.add_visualization_2d(self.intrinsics, self.dist_coeffs, bbox_annot_builder, width, height)
                bbox_annot = bbox_annot_builder.build(
                    color_msg.getTimestamp(), color_msg.getSequenceNum()
                )

                measurement_annot_builder.draw_text(
                    text=f"Length: {l:.2f}m, Width: {w:.2f}m, Height: {h:.2f}m",
                    position=(0.05, 0.1),
                    color=(0, 0, 0, 1), # black
                    background_color=(1, 1, 1, 0.7), # white with 70% opacity 
                    size=16
                )
                measurement_annot = measurement_annot_builder.build(
                    color_msg.getTimestamp(), color_msg.getSequenceNum()
                )
            else:
                bbox_annot = dai.ImgAnnotations()
                bbox_annot.setTimestamp(color_msg.getTimestamp())
                bbox_annot.setSequenceNum(color_msg.getSequenceNum())

                measurement_annot_builder.draw_text(
                    text="No box detected",
                    position=(0.05, 0.1),
                    color=(0, 0, 0, 1), # black
                    background_color=(1, 1, 1, 0.7), # white with 70% opacity 
                    size=16
                ) 
                measurement_annot = measurement_annot_builder.build(
                    color_msg.getTimestamp(), color_msg.getSequenceNum()
                )

            self.annotation_output.send(bbox_annot)
            self.passthrough.send(color_msg)
            self.measurements_output.send(measurement_annot)
