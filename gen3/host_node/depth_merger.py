import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended, ImgDetectionsExtended

from .transform import img_detection_to_points, img_detections_to_points, points_to_spatial_img_detection, points_to_spatial_img_detections


class DepthMerger(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    
    def build(self, output_2d: dai.Node.Output, output_depth: dai.Node.Output) -> "DepthMerger":
        self.link_args(output_2d, output_depth)
        return self
        
    def process(self, message_2d: dai.Buffer, stereo_depth: dai.ImgFrame) -> None:
        points_2d = self._transform(message_2d)
        points_3d = self._merge(points_2d, stereo_depth)
        message_3d = self._detransform(message_2d, points_3d)
        message_3d.setTimestamp(message_2d.getTimestamp())
        message_3d.setSequenceNum(message_2d.getSequenceNum())
        self.out.send(message_3d)

    def _transform(self, message_2d: dai.Buffer) -> dict[str, dai.Point2f] | list[dict[str, dai.Point2f]]:
        if isinstance(message_2d, dai.ImgDetection):
            return img_detection_to_points(message_2d)
        elif isinstance(message_2d, dai.ImgDetections):
            return img_detections_to_points(message_2d)
        elif isinstance(message_2d, ImgDetectionExtended):
            return img_detection_to_points(message_2d)
        elif isinstance(message_2d, ImgDetectionsExtended):
            return img_detections_to_points(message_2d)
        else:
            raise ValueError(f"Unknown message type: {type(message_2d)}")

    def _merge(self, points: list[dict[str, dai.Point2f]] | dict[str, dai.Point2f], stereo_depth: dai.ImgFrame) -> dict[str, dai.Point3f] | list[dict[str, dai.Point3f]]:
        stereo_depth_frame = stereo_depth.getFrame()
        x_len = stereo_depth_frame.shape[1]
        y_len = stereo_depth_frame.shape[0]
        if not isinstance(points, list):
            points = [points]
        object_points_3d = []
        for property_points in points:
            points_3d: dict[str, dai.Point3d] = {}
            for point_key in property_points.keys(): #TODO: try to leverage numpy, ideally avoid loop so its faster
                point_2d = property_points[point_key]
                point_3d = dai.Point3f()
                point_3d.x = point_2d.x
                point_3d.y = point_2d.y
                absolute_x = int(point_2d.x * x_len)
                absolute_y = int(point_2d.y * y_len)
                point_3d.z = stereo_depth_frame[absolute_y, absolute_x]
                points_3d[point_key] = point_3d
                object_points_3d.append(points_3d)
        if not isinstance(points, list):
            return object_points_3d[0]
        return object_points_3d
    
    def _detransform(self, message_2d: dai.Buffer, points: dict[str, dai.Point3f] | list[dict[str, dai.Point3f]]) -> dai.Buffer:
        if isinstance(message_2d, dai.ImgDetection):
            return points_to_spatial_img_detection(points, 0, 0)
        elif isinstance(message_2d, dai.ImgDetections):
            return points_to_spatial_img_detections(points, [], [])
        elif isinstance(message_2d, ImgDetectionExtended):
            return points_to_spatial_img_detection(points, 0, 0)
        elif isinstance(message_2d, ImgDetectionsExtended):
            return points_to_spatial_img_detections(points, [], [])
        else:
            raise ValueError(f"Unknown message type: {type(message_2d)}")
