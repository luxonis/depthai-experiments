import depthai as dai
import numpy as np
from parsing.messages import Keypoints, Keypoint

KEYPOINTS = {
    "nose": 0,
    "right_eye": 1,
    "left_eye": 2,
    "right_ear": 3,
    "left_ear": 4,
    "right_shoulder": 5,
    "left_shoulder": 6,
    "right_elbow": 7,
    "left_elbow": 8,
    "right_wrist": 9,
    "left_wrist": 10,
    "right_hip": 11,
    "left_hip": 12,
    "right_knee": 13,
    "left_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}

COLORS = {
    "torso": [102, 204, 0],  # Green
    "right_arm": [0, 102, 204],  # Orange
    "left_arm": [204, 0, 102],  # Magenta
    "right_leg": [255, 128, 0],  # Blue
    "left_leg": [204, 0, 204],  # Pink
    "head": [0, 204, 204]  # Yellow
}

# (Keypoint, Keypoint, Body Part)
PAIRS = [
    ("right_wrist", "right_elbow", "right_arm"),
    ("right_elbow", "right_shoulder", "right_arm"),
    ("right_shoulder", "left_shoulder", "torso"),
    ("right_shoulder", "right_ear", "head"),
    ("left_shoulder", "left_ear", "head"),
    ("left_ear", "left_eye", "head"),
    ("right_ear", "right_eye", "head"),
    ("right_eye", "left_eye", "head"),
    ("left_eye", "nose", "head"),
    ("right_eye", "nose", "head"),
    ("left_shoulder", "left_elbow", "left_arm"),
    ("left_elbow", "left_wrist", "left_arm"),
    ("right_shoulder", "right_hip", "torso"),
    ("left_shoulder", "left_hip", "torso"),
    ("right_hip", "left_hip", "torso"),
    ("right_hip", "right_knee", "right_leg"),
    ("right_knee", "right_ankle", "right_leg"),
    ("left_hip", "left_knee", "left_leg"),
    ("left_knee", "left_ankle", "left_leg")
]

class HumanPose(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output_keypts = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageAnnotations, True)])
        self.output_pose = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageAnnotations, True)])
        self._confidence_threshold = 0.5

    def build(self, keypoints: dai.Node.Output) -> "HumanPose":
        self.link_args(keypoints)
        return self
    
    def setConfidenceThreshold(self, threshold: float) -> None:
        self._confidence_threshold = threshold

    def process(self, keypoints: dai.Buffer) -> None:
        assert(isinstance(keypoints, Keypoints))

        keypts = keypoints.keypoints
        if not keypts:
            return

        keypt_annotations = dai.VectorCircleAnnotation()
        connected_keypts = dai.VectorPointsAnnotation()
        drawn_keypts: list[int] = []
        for pair in PAIRS:
            start_keypt_index = KEYPOINTS[pair[0]]
            start_keypt = keypts[start_keypt_index]
            end_keypt_index = KEYPOINTS[pair[1]]
            end_keypt = keypts[end_keypt_index]
            if start_keypt.confidence < self._confidence_threshold or end_keypt.confidence < self._confidence_threshold:
                continue

            body_part = pair[2]
            body_part_color = self._get_color(body_part)
            
            body_part_line = self._get_body_part_annotation(start_keypt, end_keypt, body_part_color)
            connected_keypts.append(body_part_line)
            if start_keypt_index not in drawn_keypts:
                keypt_annotations.append(self._get_keypt_annotation(start_keypt.x, start_keypt.y, body_part_color))
                drawn_keypts.append(start_keypt_index)
            if end_keypt_index not in drawn_keypts:
                keypt_annotations.append(self._get_keypt_annotation(end_keypt.x, end_keypt.y, body_part_color))
                drawn_keypts.append(end_keypt_index)
                
        self._send_output_keypts(keypt_annotations, keypoints)
        self._send_output_pose(connected_keypts, keypoints)

    def _get_color(self, body_part, alpha=0.80) -> dai.Color:
        color_array = np.array(COLORS[body_part]) / 255
        return dai.Color(color_array[2], color_array[1], color_array[0], alpha)

    def _get_body_part_annotation(self, start_keypt: Keypoint, end_keypt: Keypoint, color: dai.Color) -> dai.PointsAnnotation:
        connected_keypt = dai.PointsAnnotation()
        connected_keypt.points = dai.VectorPoint2f([dai.Point2f(start_keypt.x, start_keypt.y), dai.Point2f(end_keypt.x, end_keypt.y)])
        connected_keypt.fillColor = color
        connected_keypt.outlineColor = color
        connected_keypt.outlineColors = dai.VectorColor([color, color])
        connected_keypt.thickness = 5
        connected_keypt.type = dai.PointsAnnotationType.LINE_LOOP
        return connected_keypt

    def _get_keypt_annotation(self, x: float, y: float, color: dai.Color) -> dai.CircleAnnotation:
        keypt = dai.CircleAnnotation()
        keypt.position = dai.Point2f(x, y)
        keypt.diameter = 0.02
        keypt.fillColor = color
        return keypt

    def _send_output_keypts(self, keypt_annotations: dai.VectorCircleAnnotation, keypoints: Keypoints):
        keypts_annotation = dai.ImageAnnotation()
        keypts_annotation.circles = keypt_annotations
        self.output_keypts.send(self._get_annotations(keypts_annotation, keypoints))

    def _send_output_pose(self, connected_keypts: dai.PointsAnnotation, keypoints: Keypoints):
        pose_annotation = dai.ImageAnnotation()
        pose_annotation.points = connected_keypts
        self.output_pose.send(self._get_annotations(pose_annotation, keypoints))

    def _get_annotations(self, annotation: dai.ImageAnnotation, keypoints: Keypoints) -> dai.ImageAnnotations:
        vector_image_annotation = dai.VectorImageAnnotation([annotation])
        annotations = dai.ImageAnnotations(vector_image_annotation)
        annotations.setSequenceNum(keypoints.getSequenceNum())
        annotations.setTimestamp(keypoints.getTimestamp())
        return annotations
