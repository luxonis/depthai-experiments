#!/usr/bin/env python3

import depthai as dai
import av
import cv2
import time
# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
videoEnc.setLossless(True)
camRgb.video.link(videoEnc.input)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("lossless")
videoEnc.bitstream.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="lossless", maxSize=1, blocking=False)
    codec = av.CodecContext.create("mjpeg", "r")
    while True:
        data = q.get().getData()  # Blocking call, will wait until new data has arrived
        start = time.perf_counter()
        packets = codec.parse(data)
        for packet in packets:
            frames = codec.decode(packet)
            if frames:
                frame = frames[0].to_ndarray(format='bgr24')
                print(f"AV decode frame in {(time.perf_counter() - start) * 1000} milliseconds")
                cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
