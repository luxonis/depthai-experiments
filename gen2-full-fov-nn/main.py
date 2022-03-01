#!/usr/bin/env python3

import cv2
import depthai as dai
import blobconverter
import numpy as np

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.createColorCamera()
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setInterleaved(False)
camRgb.setIspScale(1,5) # 4032x3040 -> 812x608

xoutIsp = pipeline.createXLinkOut()
xoutIsp.setStreamName("isp")
camRgb.isp.link(xoutIsp.input)

# Use ImageManip to resize to 300x300 and convert YUV420 -> RGB
manip = pipeline.createImageManip()
manip.setMaxOutputFrameSize(270000) # 300x300x3
manip.initialConfig.setResizeThumbnail(300, 300)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p) # needed for NN
camRgb.isp.link(manip.inputImage)

# NN to demonstrate how to run inference on full FOV frames
nn = pipeline.createMobileNetDetectionNetwork()
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(str(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6)))
manip.out.link(nn.input)

xoutNn = pipeline.createXLinkOut()
xoutNn.setStreamName("nn")
nn.out.link(xoutNn.input)

xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
nn.passthrough.link(xoutRgb.input)


with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name='rgb')
    qNn = device.getOutputQueue(name='nn')
    qIsp = device.getOutputQueue(name='isp')

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.imshow(name, frame)

    while True:
        if qNn.has():
            dets = qNn.get().detections
            frame = qRgb.get()
            f = frame.getCvFrame()
            displayFrame("rgb", f, dets)
        if qIsp.has():
            frame = qIsp.get()
            f = frame.getCvFrame()
            cv2.putText(f, str(f.shape), (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255))
            cv2.imshow("isp", f)

        if cv2.waitKey(1) == ord('q'):
            break
