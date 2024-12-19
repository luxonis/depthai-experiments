import depthai as dai
import numpy as np
from depthai_nodes.ml.helpers.constants import TEXT_COLOR
from depthai_nodes.ml.messages import ImgDetectionsExtended, ImgDetectionExtended, Keypoint, Keypoints
from typing import List

class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        
        self.input = self.createInput()
        self.out_detections = self.createOutput()
        
    def run(self):
        while self.isRunning():
            message_group = self.input.get()
            detections_list: List[dai.ImgDetection] = message_group["detections"].detections
            keypoints_msg_list: List[Keypoints] = message_group["recognitions"].recognitions

            img_detections_exteded = ImgDetectionsExtended()

            annotations = dai.ImgAnnotations()  # custom annotations for drawing lines between keypoints
            annotation = dai.ImgAnnotation()

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
                    slope_x = detection.xmax - detection.xmin
                    slope_y = detection.ymax - detection.ymin
                    new_keypoints = []
                    xs = []
                    ys = []
                    for kp in keypoints_msg.keypoints:
                        new_kp = Keypoint()
                        new_kp.x = min(max(detection.xmin + slope_x * kp.x, 0.0), 1.0)
                        new_kp.y = min(max(detection.ymin + slope_y * kp.y, 0.0), 1.0)
                        xs.append(new_kp.x)
                        ys.append(new_kp.y)
                        new_kp.z = kp.z
                        new_kp.confidence = kp.confidence
                        new_keypoints.append(new_kp)
                    img_detection_extended.keypoints = new_keypoints
                
                img_detections_exteded.detections.append(img_detection_extended)
            
            img_detections_exteded.setTimestamp(message_group["recognitions"].getTimestamp())
            img_detections_exteded.transformation = message_group["detections"].getTransformation()

            self.out_detections.send(img_detections_exteded)
            # self.out_keypoints.send(annotations)                

            # white_frame = np.ones(frame.shape, dtype=np.uint8) * 255
            # annotation = dai.ImgAnnotation()
            # img_annotations = dai.ImgAnnotations()
            # for i, recognition in enumerate(recognitions_message.recognitions):
            #     detection = detections_list[i]
            #     points = detection.rotated_rect.getPoints()
                
            #     text_annotation = dai.TextAnnotation()
            #     text_annotation.position = points[3]
            #     text_annotation.text = "".join(recognition.classes)
            #     text_annotation.fontSize = (points[3].y - points[0].y) * white_frame.shape[0]
            #     text_annotation.textColor = TEXT_COLOR
            #     annotation.texts.append(text_annotation)
            
            # img_annotations.annotations.append(annotation)
            # img_annotations.setTimestamp(recognitions_message.getTimestamp())
            # output_frame = dai.ImgFrame()
            
            # self.white_frame_output.send(output_frame.setCvFrame(white_frame, dai.ImgFrame.Type.NV12))
            # self.text_annotations_output.send(img_annotations)
                