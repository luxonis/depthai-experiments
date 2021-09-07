#!/usr/bin/env python3

import depthai as dai
import subprocess as sp
from os import name as osName

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.createColorCamera()
videoEnc = pipeline.createVideoEncoder()
xout = pipeline.createXLinkOut()

xout.setStreamName("h265")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
videoEnc.setDefaultProfilePreset(3840, 2160, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)

# Linking
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

width, height = 720, 500
cmd_out = ["ffplay", "-i", "-", "-x", str(width), "-y", str(height)]
if osName == "nt":  # Running on Windows
    cmd_out = ["cmd", "/c"] + cmd_out

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)

    proc = sp.Popen(cmd_out, stdin=sp.PIPE)  # Start the ffplay process
    try:
        while True:
            h265Packet = q.get()  # Blocking call, will wait until a new data has arrived
            h265Packet.getData().tofile(proc.stdin)  # Appends the packet data to the process pipe
    except:
        pass

    proc.stdin.close()
