import cv2
import numpy as np

MIN_THRESHOLD = 15.  # Degrees in yaw/pitch/roll to be considered as head movement

bg_color = (0, 0, 0)
color = (255, 255, 255)
text_type = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA


def putText(frame, text, coords, size=0.6, bold=False):
    mult = 2 if bold else 1
    cv2.putText(frame, text, coords, text_type, size, bg_color, 3 * mult, line_type)
    cv2.putText(frame, text, coords, text_type, size, color, 1 * mult, line_type)


def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def decode_pose(yaw, pitch, roll, face_bbox, frame):
    """
    pitch > 0 Head down, < 0 look up
    yaw > 0 Turn right < 0 Turn left
    roll > 0 Tilt right, < 0 Tilt left
    """
    putText(frame, "pitch:{:.0f}, yaw:{:.0f}, roll:{:.0f}".format(pitch, yaw, roll),
            (face_bbox[0] + 10 - 15, face_bbox[1] - 15))

    vals = np.array([abs(pitch), abs(yaw), abs(roll)])
    max_index = np.argmax(vals)

    if vals[max_index] < MIN_THRESHOLD: return

    if max_index == 0:
        if pitch > 0:
            txt = "Look down"
        else:
            txt = "Look up"
    elif max_index == 1:
        if yaw > 0:
            txt = "Turn right"
        else:
            txt = "Turn left"
    elif max_index == 2:
        if roll > 0:
            txt = "Tilt right"
        else:
            txt = "Tilt left"
    putText(frame, txt, (face_bbox[0] + 10, face_bbox[1] + 30), size=1, bold=True)
