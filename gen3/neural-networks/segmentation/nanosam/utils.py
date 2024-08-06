import numpy as np

def generate_overlay(frame, mask, weight=0.5):
    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
    frame = (
        frame * (1 - mask) + (frame * weight + np.array([255, 0, 255]) * weight) * mask
    ).astype(np.uint8)

    return frame

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
