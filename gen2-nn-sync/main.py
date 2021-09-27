#!/usr/bin/env python3
import queue
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6)))
cam_rgb.preview.link(detection_nn.input)

landmarks_nn = pipeline.createNeuralNetwork()
landmarks_nn.setBlobPath(str(blobconverter.from_zoo(name="landmarks-regression-retail-0009", shaves=6)))

# Create outputs
xin_rgb = pipeline.createXLinkIn()
xin_rgb.setStreamName("land_in")
xin_rgb.out.link(landmarks_nn.input)

# Create outputs
xout_frame = pipeline.createXLinkOut()
xout_frame.setStreamName("det_frame")
cam_rgb.preview.link(xout_frame.input)

xout_det = pipeline.createXLinkOut()
xout_det.setStreamName("det_nn")
detection_nn.out.link(xout_det.input)

xout_land = pipeline.createXLinkOut()
xout_land.setStreamName("land_nn")
landmarks_nn.out.link(xout_land.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_frame = device.getOutputQueue(name="det_frame", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="det_nn", maxSize=4, blocking=False)
    land_in = device.getInputQueue(name="land_in", maxSize=4, blocking=False)
    q_land = device.getOutputQueue(name="land_nn", maxSize=4, blocking=False)

    face_q = queue.Queue()

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frame_norm(frame, bbox):
        return (np.clip(np.array(bbox), 0, 1) * np.array(frame.shape[:2] * (len(bbox) // 2))[::-1]).astype(int)


    while True:
        while q_det.has():
            in_frame = q_frame.get()
            shape = (3, in_frame.getHeight(), in_frame.getWidth())
            det_frame = in_frame.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            det_frame = np.ascontiguousarray(det_frame)
            # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
            bboxes = np.array(q_det.get().getFirstLayerFp16())
            # take only the results before -1 digit
            bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
            # transform the 1D array into Nx7 matrix
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            # filter out the results which confidence less than a defined threshold
            bboxes = bboxes[bboxes[:, 2] > 0.5][:, 3:7]
            for raw_bbox in bboxes:
                bbox = frame_norm(det_frame, raw_bbox)
                cv2.rectangle(det_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                face_frame = det_frame[
                    bbox[1]:bbox[3],
                    bbox[0]:bbox[2]
                ]

                nn_data = dai.NNData()
                nn_data.setLayer("0", to_planar(face_frame, (48, 48)))
                land_in.send(nn_data)
                face_q.put(face_frame)
                cv2.imshow("rgb", det_frame)

            while q_land.has():
                face_frame = face_q.get()
                out = frame_norm(face_frame, q_land.get().getFirstLayerFp16())
                cv2.circle(face_frame, tuple(out[:2]), 1, (255, 255, 0))  # Right eye
                cv2.circle(face_frame, tuple(out[2:4]), 1, (255, 255, 0))  # Left eye
                cv2.circle(face_frame, tuple(out[4:6]), 1, (255, 255, 0))  # Nose
                cv2.circle(face_frame, tuple(out[6:8]), 1, (255, 255, 0))  # Right mouth
                cv2.circle(face_frame, tuple(out[8:]), 1, (255, 255, 0))  # Left mouth
                cv2.imshow("face", face_frame)

        if cv2.waitKey(1) == ord('q'):
            break
