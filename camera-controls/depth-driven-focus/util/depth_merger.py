import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended, ImgDetectionsExtended

from .host_spatials_calc import HostSpatialsCalc


class DepthMerger(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.SpatialImgDetections, True)
            ]
        )

    def build(
        self,
        output_2d: dai.Node.Output,
        output_depth: dai.Node.Output,
        calib_data: dai.CalibrationHandler,
        depth_alignment_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
    ) -> "DepthMerger":
        self.link_args(output_2d, output_depth)
        self.host_spatials_calc = HostSpatialsCalc(calib_data, depth_alignment_socket)
        return self

    def process(self, message_2d: dai.Buffer, depth: dai.ImgFrame) -> None:
        spatial_dets = self._transform(message_2d, depth)
        self.output.send(spatial_dets)

    def _transform(
        self, message_2d: dai.Buffer, depth: dai.ImgFrame
    ) -> dai.SpatialImgDetections | dai.SpatialImgDetection:
        if isinstance(message_2d, dai.ImgDetection):
            return self._detection_to_spatial(message_2d, depth)
        elif isinstance(message_2d, dai.ImgDetections):
            return self._detections_to_spatial(message_2d, depth)
        elif isinstance(message_2d, ImgDetectionExtended):
            return self._detection_to_spatial(message_2d, depth)
        elif isinstance(message_2d, ImgDetectionsExtended):
            return self._detections_to_spatial(message_2d, depth)
        else:
            raise ValueError(f"Unknown message type: {type(message_2d)}")

    def _detection_to_spatial(
        self, detection: dai.ImgDetection | ImgDetectionExtended, depth: dai.ImgFrame
    ) -> dai.SpatialImgDetection:
        depth_frame = depth.getCvFrame()
        x_len = depth_frame.shape[1]
        y_len = depth_frame.shape[0]
        if isinstance(detection, ImgDetectionExtended):
            xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()
        else:
            xmin, ymin, xmax, ymax = (
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
            )
        roi = [
            self._get_index(xmin, x_len),
            self._get_index(ymin, y_len),
            self._get_index(xmax, x_len),
            self._get_index(ymax, y_len),
        ]

        spatials = self.host_spatials_calc.calc_spatials(depth, roi)

        spatial_img_detection = dai.SpatialImgDetection()
        spatial_img_detection.xmin = xmin
        spatial_img_detection.ymin = ymin
        spatial_img_detection.xmax = xmax
        spatial_img_detection.ymax = ymax
        spatial_img_detection.spatialCoordinates = dai.Point3f(
            spatials["x"], spatials["y"], spatials["z"]
        )

        spatial_img_detection.confidence = detection.confidence
        spatial_img_detection.label = max(0, detection.label)
        return spatial_img_detection

    def _detections_to_spatial(
        self, detections: dai.ImgDetections | ImgDetectionsExtended, depth: dai.ImgFrame
    ) -> dai.SpatialImgDetections:
        new_dets = dai.SpatialImgDetections()
        new_dets.detections = [
            self._detection_to_spatial(d, depth) for d in detections.detections
        ]
        new_dets.setSequenceNum(detections.getSequenceNum())
        new_dets.setTimestamp(detections.getTimestamp())
        return new_dets

    def _get_index(self, relative_coord: float, dimension_len: int) -> int:
        bounded_coord = min(1, relative_coord)
        return max(0, int(bounded_coord * dimension_len) - 1)
