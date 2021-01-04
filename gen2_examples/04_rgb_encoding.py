#!/usr/bin/env python3

import subprocess
import depthai as dai

pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setCamId(0)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

videoEncoder = pipeline.createVideoEncoder()
videoEncoder.setDefaultProfilePreset(3840, 2160, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam.video.link(videoEncoder.input)

videoOut = pipeline.createXLinkOut()
videoOut.setStreamName('h265')
videoEncoder.bitstream.link(videoOut.input)

device = dai.Device(pipeline)
device.startPipeline()

q = device.getOutputQueue(name="h265")

with open('video.h265','wb') as videoFile:
    print("Press Ctrl+C to stop encoding...")
    try:
        while True:
            h264Packet = q.get()
            h264Packet.getData().tofile(videoFile)
    except KeyboardInterrupt:
        pass

print("Converting stream file (.h265) into a video file (.mp4)...")
subprocess.check_call("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4".split())
print("Conversion successful, check video.mp4")
