import depthai as dai
from .box_estimator import BoxEstimator
from .projector_3d import PointCloudFromRGBD
from .img_annotation_helper import AnnotationHelper


class BoxMeasurement(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
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

    def build(
        self,
        color: dai.Node.Output,
        depth: dai.Node.Output,
        cam_intrinsics: list,
        shape: tuple[int, int],
        max_dist: float,
        min_box_size: float,
    ) -> "BoxMeasurement":

        self.intrinsics = cam_intrinsics
        self.min_box_size = min_box_size

        self.pcl_converter = PointCloudFromRGBD(cam_intrinsics, shape[0], shape[1])
        self.box_estimator = BoxEstimator(max_dist)
        self.link_args(color, depth)
        return self

    def create_text_annot(self, text, pos):
        txt_annot = dai.TextAnnotation()
        txt_annot.fontSize = 10
        txt_annot.backgroundColor = dai.Color(0, 1, 0, 1)
        txt_annot.textColor = dai.Color(1, 1, 1, 1)
        txt_annot.position = dai.Point2f(*pos)
        txt_annot.text = text
        return txt_annot

    def process(self, color: dai.ImgFrame, depth: dai.ImgFrame) -> None:
        color_frame = color.getCvFrame()
        depth_frame = depth.getFrame()
        pointcloud = self.pcl_converter.rgbd_to_projection(depth_frame, color_frame)

        l, w, h = self.box_estimator.process_pcl(pointcloud)

        bbox_annot_builder = AnnotationHelper()
        measurement_annot_builder = AnnotationHelper()

        if l * w * h > self.min_box_size:
            # Create ImgAnnotations and draw lines
            height, width, _ = color_frame.shape
            self.box_estimator.add_visualization_2d(self.intrinsics, bbox_annot_builder, width, height)
            bbox_annot = bbox_annot_builder.build(
                color.getTimestamp(), color.getSequenceNum()
            )

            measurement_annot_builder.draw_text(
                text=f"Length: {l:.2f}m, Width: {w:.2f}m, Height: {h:.2f}m",
                position=(0.05, 0.1),
                color=(0, 0, 0, 1), # black
                background_color=(1, 1, 1, 0.7), # white with 70% opacity 
                size=16
            )
            measurement_annot = measurement_annot_builder.build(
                color.getTimestamp(), color.getSequenceNum()
            )
        else:
            bbox_annot = dai.ImgAnnotations()
            bbox_annot.setTimestamp(color.getTimestamp())
            bbox_annot.setSequenceNum(color.getSequenceNum())

            measurement_annot_builder.draw_text(
                text="No box detected",
                position=(0.05, 0.1),
                color=(0, 0, 0, 1), # black
                background_color=(1, 1, 1, 0.7), # white with 70% opacity 
                size=16
            ) 
            measurement_annot = measurement_annot_builder.build(
                color.getTimestamp(), color.getSequenceNum()
            )

        self.annotation_output.send(bbox_annot)
        self.passthrough.send(color)
        self.measurements_output.send(measurement_annot)
