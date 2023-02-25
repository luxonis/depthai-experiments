import cv2
import numpy as np

from depthai_sdk import OakCamera

NN_WIDTH, NN_HEIGHT = 300, 300


def callback(packet):
    frame = packet.frame

    for det in packet.detections:
        # Expand the bounding box
        x1, y1, x2, y2 = det.top_left[0], det.top_left[1], det.bottom_right[0], det.bottom_right[1]
        x1 *= 0.8
        y1 *= 0.8
        x2 *= 1.2
        y2 *= 1.2

        bbox = np.int0([x1, y1, x2, y2])

        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        fh, fw, fc = face.shape
        frame_h, frame_w, frame_c = frame.shape

        # Create blur mask around the face
        mask = np.zeros((frame_h, frame_w), np.uint8)
        polygon = cv2.ellipse2Poly((bbox[0] + int(fw / 2), bbox[1] + int(fh / 2)), (int(fw / 2), int(fh / 2)), 0, 0,
                                   360, delta=1)
        cv2.fillConvexPoly(mask, polygon, 255)

        frame_copy = frame.copy()
        frame_copy = cv2.blur(frame_copy, (80, 80))
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=background_mask)
        # Blur the face
        frame = cv2.add(background, face_extracted)

    cv2.imshow('Face blurring', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', fps=30)
    det_nn = oak.create_nn('face-detection-retail-0004', color)
    det_nn.config_nn(resize_mode='crop')

    oak.callback(det_nn.out.passthrough, callback=callback)
    oak.start(blocking=True)
