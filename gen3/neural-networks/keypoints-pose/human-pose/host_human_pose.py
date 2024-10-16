import depthai as dai
import cv2
from parsing.messages import Keypoints

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
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        self._confidence_threshold = 0.5

    def build(self, preview: dai.Node.Output, keypoints: dai.Node.Output) -> "HumanPose":
        self.link_args(preview, keypoints)
        return self
    
    def setConfidenceThreshold(self, threshold: float) -> None:
        self._confidence_threshold = threshold

    # Preview is actually type dai.ImgFrame here
    def process(self, preview: dai.Buffer, keypoints: dai.Buffer) -> None:
        assert(isinstance(preview, dai.ImgFrame))
        assert(isinstance(keypoints, Keypoints))
        frame = preview.getCvFrame()

        keypts = keypoints.keypoints
        if keypts:
            def scale(point: tuple[int, int]) -> tuple[int, int]:
                return int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
            
            drawn_keypts: list[int] = []
            for pair in PAIRS:
                start_keypt_index = KEYPOINTS[pair[0]]
                start_keypt = keypts[start_keypt_index]
                end_keypt_index = KEYPOINTS[pair[1]]
                end_keypt = keypts[end_keypt_index]
                if start_keypt.confidence < self._confidence_threshold or end_keypt.confidence < self._confidence_threshold:
                    continue

                body_part = pair[2]
                cv2.line(frame, scale((start_keypt.x, start_keypt.y)), scale((end_keypt.x, end_keypt.y)), COLORS[body_part], 2, cv2.LINE_AA)
                if start_keypt_index not in drawn_keypts:
                    cv2.circle(frame, scale((start_keypt.x, start_keypt.y)), 5, COLORS[body_part], -1, cv2.LINE_AA)
                    drawn_keypts.append(start_keypt_index)
                if end_keypt_index not in drawn_keypts:
                    cv2.circle(frame, scale((end_keypt.x, end_keypt.y)), 5, COLORS[body_part], -1, cv2.LINE_AA)
                    drawn_keypts.append(end_keypt_index)
                

        preview.setCvFrame(frame, dai.ImgFrame.Type.BGR888i)
        self.output.send(preview)
