#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import blobconverter
from utility import *

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

NN_SIZE = (300,300)

# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(NN_SIZE)
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setIspScale(1,3) # You don't need to downscale (4k -> 720P) video frames

xoutFrames = pipeline.create(dai.node.XLinkOut)
xoutFrames.setStreamName("frames")
camRgb.video.link(xoutFrames.input)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
camRgb.preview.link(nn.input)

passthroughOut = pipeline.create(dai.node.XLinkOut)
passthroughOut.setStreamName("pass")
nn.passthrough.link(passthroughOut.input)

nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qFrames = device.getOutputQueue(name="frames")
    qPass = device.getOutputQueue(name="pass")
    qDet = device.getOutputQueue(name="nn")

    detections = []
    fps = FPSHandler()
    text = TextHelper()

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, NN_SIZE, bbox):
        # Check difference in aspect ratio and apply correction to BBs
        ar_diff = NN_SIZE[0] / NN_SIZE[0] - frame.shape[0] / frame.shape[1]
        sel = 0 if 0 < ar_diff else 1
        bbox[sel::2] *= 1-abs(ar_diff)
        bbox[sel::2] += abs(ar_diff)/2
        # Normalize bounding boxes
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(bbox, 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, NN_SIZE, np.array([detection.xmin, detection.ymin, detection.xmax, detection.ymax]))
            text.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20))
            text.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40))
            text.rectangle(frame, bbox)
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        frame = qFrames.get().getCvFrame()

        inDet = qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections
            fps.next_iter()

        inPass = qPass.tryGet()
        if inPass is not None:
            displayFrame('Passthrough', inPass.getCvFrame())

        # If the frame is available, draw bounding boxes on it and show the frame
        text.putText(frame, "NN fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4))
        displayFrame("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
