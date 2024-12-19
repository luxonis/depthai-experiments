import depthai as dai
import numpy as np
from depthai_nodes.ml.helpers.constants import OUTLINE_COLOR
from depthai_nodes.ml.messages import ImgDetectionsExtended, ImgDetectionExtended, Keypoint, Keypoints
from typing import List

class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self, connection_pairs: List[List[int]]) -> None:
        super().__init__()
        
        self.input = self.createInput()
        self.out_detections = self.createOutput()
        self.out_pose_annotations = self.createOutput()
        self.connection_pairs = connection_pairs
        
    def run(self):
        while self.isRunning():
            message_group = self.input.get()
            detections_list: List[dai.ImgDetection] = message_group["detections"].detections
            keypoints_msg_list: List[Keypoints] = message_group["recognitions"].recognitions

            img_detections_exteded = ImgDetectionsExtended()

            annotations = dai.ImgAnnotations()  # custom annotations for drawing lines between keypoints
            annotation = dai.ImgAnnotation()

            padding = 0.1

            for ix, detection in enumerate(detections_list):
                img_detection_extended = ImgDetectionExtended()
                center_x = detection.xmin + (detection.xmax - detection.xmin) / 2
                center_y = detection.ymin + (detection.ymax - detection.ymin) / 2
                width = detection.xmax - detection.xmin
                height = detection.ymax - detection.ymin
                angle = 0
                img_detection_extended.rotated_rect = (center_x, center_y, width, height, angle)
                img_detection_extended.label = detection.label
                img_detection_extended.confidence = detection.confidence

                if keypoints_msg_list is not None:
                    keypoints_msg = keypoints_msg_list[ix]
                    slope_x = (detection.xmax + padding) - (detection.xmin - padding)
                    slope_y = (detection.ymax + padding) - (detection.ymin - padding)
                    new_keypoints = []
                    xs = []
                    ys = []
                    for kp in keypoints_msg.keypoints:
                        new_kp = Keypoint()
                        new_kp.x = min(max(detection.xmin - padding + slope_x * kp.x, 0.0), 1.0)
                        new_kp.y = min(max(detection.ymin - padding + slope_y * kp.y, 0.0), 1.0)
                        xs.append(new_kp.x)
                        ys.append(new_kp.y)
                        new_kp.z = kp.z
                        new_kp.confidence = kp.confidence
                        new_keypoints.append(new_kp)
                    img_detection_extended.keypoints = new_keypoints

                    for connection in self.connection_pairs:
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
                annotations.setTimestamp(message_group["detections"].getTimestamp())
            
            img_detections_exteded.setTimestamp(message_group["detections"].getTimestamp())
            img_detections_exteded.transformation = message_group["detections"].getTransformation()

            self.out_detections.send(img_detections_exteded)
            self.out_pose_annotations.send(annotations)
                