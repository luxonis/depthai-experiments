#!/usr/bin/env python3

import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Properties
videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.video.link(videoEnc.input)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("mjpeg")
videoEnc.bitstream.link(xout.input)

with dai.Device(pipeline) as device:
    # Output queue will be used to get the encoded stream
    q = device.getOutputQueue("mjpeg")

    while True:
        # Receive encoded frame
        imgFrame: dai.ImgFrame = q.get()  # Blocking call, will wait until new data has arrived
        # Decode the MJPEG frame
        frame = cv2.imdecode(imgFrame.getData(), cv2.IMREAD_COLOR)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
