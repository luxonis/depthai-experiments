import depthai as dai
from depthai_nodes.ml.messages import Keypoints, ImgDetectionsExtended, ImgDetectionExtended, Keypoint
from depthai_nodes.ml.helpers.constants import OUTLINE_COLOR
from typing import List

class CustomVisualizer(dai.node.ThreadedHostNode):
    def __init__(self, connection_pairs: List[List[int]]) -> None:
        super().__init__()
        self.input = self.createInput()
        self.out_detections = self.createOutput()
        self.out_keypoints = self.createOutput()
        self.connections_pairs = connection_pairs

    def run(self):
        while self.isRunning():
            try:
                message_group: dai.MessageGroup = self.input.get()
            except dai.MessageQueue.QueueException:
                break

            detections_msg: dai.ImgDetections = message_group["detections"]
            keypoints_msg: Keypoints = message_group["keypoints"]
            detections = detections_msg.detections

            img_detections_exteded = ImgDetectionsExtended()

            annotations = dai.ImgAnnotations()  # custom annotations for drawing lines between keypoints
            annotation = dai.ImgAnnotation()

            for detection in detections:
                img_detection_extended = ImgDetectionExtended()
                center_x = detection.xmin + (detection.xmax - detection.xmin) / 2
                center_y = detection.ymin + (detection.ymax - detection.ymin) / 2
                width = detection.xmax - detection.xmin
                height = detection.ymax - detection.ymin
                angle = 0
                img_detection_extended.rotated_rect = (center_x, center_y, width, height, angle)
                img_detection_extended.label = detection.label
                img_detection_extended.confidence = detection.confidence
                
                if keypoints_msg is not None:
                    slope_x = detection.xmax - detection.xmin
                    slope_y = detection.ymax - detection.ymin
                    new_keypoints = []
                    xs = []
                    ys = []
                    for kp in keypoints_msg.keypoints:
                        new_kp = Keypoint()
                        new_kp.x = detection.xmin + slope_x * kp.x
                        new_kp.y = detection.ymin + slope_y * kp.y
                        xs.append(new_kp.x)
                        ys.append(new_kp.y)
                        new_kp.z = kp.z
                        new_kp.confidence = kp.confidence
                        new_keypoints.append(new_kp)
                    img_detection_extended.keypoints = new_keypoints

                    for connection in self.connections_pairs:
                        pt1_idx, pt2_idx = connection
                        if pt1_idx < len(new_keypoints) and pt2_idx < len(new_keypoints):
                            x1, y1 = xs[pt1_idx], ys[pt1_idx]
                            x2, y2 = xs[pt2_idx], ys[pt2_idx]
                            pointsAnnotation = dai.PointsAnnotation()
                            pointsAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
                            pointsAnnotation.points = dai.VectorPoint2f(
                                [dai.Point2f(x=x1, y=y1, normalized=True), dai.Point2f(x=x2, y=y2, normalized=True)]
                            )
                            pointsAnnotation.outlineColor = OUTLINE_COLOR
                            pointsAnnotation.thickness = 2.0
                            annotation.points.append(pointsAnnotation)

                img_detections_exteded.detections.append(img_detection_extended)

                annotations.annotations.append(annotation)
                annotations.setTimestamp(detections_msg.getTimestamp())

            img_detections_exteded.setTimestamp(detections_msg.getTimestamp())
            img_detections_exteded.transformation = detections_msg.getTransformation()

            self.out_detections.send(img_detections_exteded)
            self.out_keypoints.send(annotations)