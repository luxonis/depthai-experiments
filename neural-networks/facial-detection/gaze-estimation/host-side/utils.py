import cv2
import numpy as np
import depthai as dai


def frame_norm(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray[np.int64]:
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    if arr.size == 0:
        return []
    return [
        val
        for channel in cv2.resize(arr, shape).transpose(2, 0, 1)
        for y_col in channel
        for val in y_col
    ]


def copy_timestamps(source: dai.Buffer, target: dai.Buffer) -> None:
    target.setSequenceNum(source.getSequenceNum())
    target.setTimestamp(source.getTimestamp())
    target.setTimestampDevice(source.getTimestampDevice())
