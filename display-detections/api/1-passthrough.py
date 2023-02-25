#!/usr/bin/env python3

import blobconverter
import depthai as dai
import numpy as np

from utility import *

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setFps(40)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
camRgb.preview.link(nn.input)

# Send passthrough frames to the host, so frames are in sync with bounding boxes
passthroughOut = pipeline.create(dai.node.XLinkOut)
passthroughOut.setStreamName("pass")
nn.passthrough.link(passthroughOut.input)

nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qPass = device.getOutputQueue(name="pass")
    qDet = device.getOutputQueue(name="nn")

    detections = []
    fps = FPSHandler()
    text = TextHelper()


    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            text.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20))
            text.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40))
            text.rectangle(frame, bbox)
        # Show the frame
        cv2.imshow(name, frame)


    while True:
        frame = qPass.get().getCvFrame()
        inDet = qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections
            fps.next_iter()

        # If the frame is available, draw bounding boxes on it and show the frame
        text.putText(frame, "NN fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4))
        displayFrame("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
