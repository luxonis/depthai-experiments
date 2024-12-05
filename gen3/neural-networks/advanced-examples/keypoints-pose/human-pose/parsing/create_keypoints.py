from typing import List, Union

import numpy as np

from .messages import Keypoint, Keypoints


def create_keypoints_message(
    keypoints: Union[np.ndarray, List[List[float]]],
    scores: Union[np.ndarray, List[float]] = None,
    confidence_threshold: float = None,
) -> Keypoints:
    """Create a DepthAI message for the keypoints.

    @param keypoints: Detected 2D or 3D keypoints of shape (N,2 or 3) meaning [...,[x, y],...] or [...,[x, y, z],...].
    @type keypoints: np.ndarray or List[List[float]]
    @param scores: Confidence scores of the detected keypoints.
    @type scores: Optional[np.ndarray or List[float]]
    @param confidence_threshold: Confidence threshold of keypoint detections.
    @type confidence_threshold: Optional[float]

    @return: Keypoints message containing the detected keypoints.
    @rtype: Keypoints

    @raise ValueError: If the keypoints are not a numpy array or list.
    @raise ValueError: If the scores are not a numpy array or list.
    @raise ValueError: If scores and keypoints do not have the same length.
    @raise ValueError: If score values are not floats.
    @raise ValueError: If score values are not between 0 and 1.
    @raise ValueError: If the confidence threshold is not a float.
    @raise ValueError: If the confidence threshold is not between 0 and 1.
    @raise ValueError: If the keypoints are not of shape (N,2 or 3).
    @raise ValueError: If the keypoints 2nd dimension is not of size E{2} or E{3}.
    """

    if not isinstance(keypoints, (np.ndarray, list)):
        raise ValueError(
            f"Keypoints should be numpy array or list, got {type(keypoints)}."
        )

    if scores is not None:
        if not isinstance(scores, (np.ndarray, list)):
            raise ValueError(
                f"Scores should be numpy array or list, got {type(scores)}."
            )

        if len(keypoints) != len(scores):
            raise ValueError(
                "Keypoints and scores should have the same length. Got {} keypoints and {} scores.".format(
                    len(keypoints), len(scores)
                )
            )

        if not all(isinstance(score, (float, np.floating)) for score in scores):
            raise ValueError("Scores should only contain float values.")
        if not all(0 <= score <= 1 for score in scores):
            scores = np.clip(scores, 0, 1)

    if confidence_threshold is not None:
        if not isinstance(confidence_threshold, float):
            raise ValueError(
                f"The confidence_threshold should be float, got {type(confidence_threshold)}."
            )

        if not (0 <= confidence_threshold <= 1):
            raise ValueError(
                f"The confidence_threshold should be between 0 and 1, got confidence_threshold {confidence_threshold}."
            )

    if len(keypoints) != 0:
        dimension = len(keypoints[0])
        if dimension != 2 and dimension != 3:
            raise ValueError(
                f"All keypoints should be of dimension 2 or 3, got dimension {dimension}."
            )

    if isinstance(keypoints, list):
        for keypoint in keypoints:
            if not isinstance(keypoint, list):
                raise ValueError(
                    f"Keypoints should be list of lists or np.array, got list of {type(keypoint)}."
                )
            if len(keypoint) != dimension:
                raise ValueError(
                    "All keypoints have to be of same dimension e.g. [x, y] or [x, y, z], got mixed inner dimensions."
                )
            for coord in keypoint:
                if not isinstance(coord, (float, np.floating)):
                    raise ValueError(
                        f"Keypoints inner list should contain only float, got {type(coord)}."
                    )

    keypoints = np.array(keypoints)
    if scores is not None:
        scores = np.array(scores)

    if len(keypoints) != 0:
        if len(keypoints.shape) != 2:
            raise ValueError(
                f"Keypoints should be of shape (N,2 or 3) got {keypoints.shape}."
            )

        use_3d = keypoints.shape[1] == 3

    keypoints_msg = Keypoints()
    points = []
    for i, keypoint in enumerate(keypoints):
        if scores is not None:
            if scores[i] < confidence_threshold:
                continue
        pt = Keypoint()
        pt.x = float(keypoint[0])
        pt.y = float(keypoint[1])
        pt.z = float(keypoint[2]) if use_3d else 0.0
        if scores is not None:
            pt.confidence = float(scores[i])
        points.append(pt)

    keypoints_msg.keypoints = points
    return keypoints_msg