#!/usr/bin/env python3

import depthai as dai
import av
import cv2
# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.H264_MAIN)
camRgb.video.link(videoEnc.input)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("h264")
videoEnc.bitstream.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="h264", maxSize=30, blocking=True)
    
    codec = av.CodecContext.create("h264", "r")
    while True:
        data = q.get().getData()  # Blocking call, will wait until new data has arrived
        packets = codec.parse(data)

        for packet in packets:
            frames = codec.decode(packet)
            if frames:
                frame = frames[0].to_ndarray(format='bgr24')
                cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
